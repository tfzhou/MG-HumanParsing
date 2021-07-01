import argparse
import json
import logging
import os
import sys
import time
import zipfile
import copy
import cv2
from PIL import Image

import numpy as np
import PIL
import thop
import torch
import torch.nn as nn

from . import datasets, decoder, network, show, transforms, visualizer, __version__


LOG = logging.getLogger(__name__)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():  # pylint: disable=too-many-statements,too-many-branches
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.eval_coco',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.cli(parser)
    decoder.cli(parser, force_complete_pose=True)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
    parser.add_argument('--detection-annotations', default=False, action='store_true')
    parser.add_argument('-n', default=0, type=int,
                        help='number of batches')
    parser.add_argument('--skip-n', default=0, type=int,
                        help='skip n batches')
    parser.add_argument('--dataset', choices=('val', 'test', 'test-dev'), default='val',
                        help='dataset to evaluate')
    parser.add_argument('--min-ann', default=0, type=int,
                        help='minimum number of truth annotations')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--long-edge', default=641, type=int,
                        help='long edge of input images. Setting to zero deactivates scaling.')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--orientation-invariant', default=False, action='store_true')
    parser.add_argument('--extended-scale', default=False, action='store_true')
    parser.add_argument('--skip-existing', default=False, action='store_true',
                        help='skip if output eval file exists already')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--write-predictions', default=False, action='store_true',
                        help='write a json and a zip file of the predictions')
    parser.add_argument('--all-images', default=False, action='store_true',
                        help='run over all images irrespective of catIds')

    group = parser.add_argument_group('logging')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    group.add_argument('--debug-images', default=False, action='store_true',
                       help='print debug messages and enable all debug images')
    group.add_argument('--log-stats', default=False, action='store_true',
                       help='enable stats logging')

    args = parser.parse_args()

    if args.debug_images:
        args.debug = True

    log_level = logging.INFO if not args.debug else logging.DEBUG
    if args.log_stats:
        # pylint: disable=import-outside-toplevel
        from pythonjsonlogger import jsonlogger
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(
            jsonlogger.JsonFormatter('(message) (levelname) (name)'))
        logging.basicConfig(handlers=[stdout_handler])
        logging.getLogger('openpifpaf').setLevel(log_level)
        logging.getLogger('openpifpaf.stats').setLevel(logging.DEBUG)
        LOG.setLevel(log_level)
    else:
        logging.basicConfig()
        logging.getLogger('openpifpaf').setLevel(log_level)
        LOG.setLevel(log_level)

    network.configure(args)
    show.configure(args)
    visualizer.configure(args)

    if args.loader_workers is None:
        args.loader_workers = max(2, args.batch_size)

    if args.dataset in ('test', 'test-dev') and not args.write_predictions and not args.debug:
        raise Exception('have to use --write-predictions for this dataset')
    if args.dataset in ('test', 'test-dev') and not args.all_images and not args.debug:
        raise Exception('have to use --all-images for this dataset')

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    return args


def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def dataloader_from_args(args):
    root = './data/CIHP'
    val_lst = os.path.join(root, 'val_id.txt')

    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ])

    val_dataset = datasets.ValidationLoader(root=root, list_path=val_lst, crop_size=473,
                                            test_transforms=test_transform)
    data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, pin_memory=args.pin_memory,
        num_workers=args.loader_workers)

    return data_loader


def main():
    args = cli()

    # skip existing?
    if args.skip_existing:
        if os.path.exists(args.output + '.stats.json'):
            print('Output file {} exists already. Exiting.'
                  ''.format(args.output + '.stats.json'))
            return
        print('Processing: {}'.format(args.checkpoint))

    data_loader = dataloader_from_args(args)
    model_cpu, _ = network.factory_from_args(args)
    model = model_cpu.to(args.device)

    confusion_matrix = np.zeros((20, 20))

    output_dir = 'cihp_global_parsing'
    os.makedirs(output_dir, exist_ok=True)
    for batch_i, (image, label, ori_size, names) in enumerate(data_loader):
        print(batch_i, '/', len(data_loader))
        ori_size = ori_size[0].numpy()
        interp = nn.Upsample(size=(np.asscalar(ori_size[0]), np.asscalar(ori_size[1])),
                             mode='bilinear', align_corners=True)
        outputs = []
        with torch.no_grad():
            image = image.cuda()
            prediction = model(image)
            prediction = interp(prediction[-1][0]).cpu().data.numpy()
            outputs.append(prediction[0, :, :, :])
        outputs = np.mean(outputs, axis=0)
        outputs = outputs.transpose(1, 2, 0)

        seg_pred = np.asarray(np.argmax(outputs, axis=2), dtype=np.uint8)

        # save
        seg_pred_out = Image.fromarray(seg_pred)
        seg_pred_out.putpalette(get_palette(20))
        seg_pred_out.save(os.path.join(output_dir, names[0] + '.png'))

        seg_gt = np.asarray(label[0].numpy(), dtype=np.int)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]

        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, 20)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = tp.sum() / pos.sum()
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()
    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    # get_confusion_matrix_plot()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IU)
    for index, IU in enumerate(IU_array):
        print('%f ', IU)


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calculate the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


if __name__ == '__main__':
    main()