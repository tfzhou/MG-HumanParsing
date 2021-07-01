import os
import cv2
import copy
import logging
import numpy as np
import os.path as osp
import scipy.io

from torch.utils import data

from .. import transforms, utils

from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class MHP(data.Dataset):
    MHP_ROOT = 'data/MHP'
    def __init__(self, root=MHP_ROOT, train=True, category_ids=None,
                 preprocess=None, target_transforms=None):

        if category_ids is None:
            category_ids = [1]
        self.category_ids = category_ids

        self.root = root
        self.split = 'train' if train else 'val'

        image_list = os.path.join(self.root, 'list', self.split + '.txt')
        with open(image_list) as f:
            self.img_ids = f.readlines()
            self.img_ids = [x.strip() for x in self.img_ids]

        self.image_folder = os.path.join(self.root, self.split, 'images')
        self.pose_folder = os.path.join(self.root, self.split, 'pose_annos')
        self.parsing_folder = os.path.join(self.root, self.split, 'parsing_annos')

        LOG.info('dataset size: {}'.format(len(self.img_ids)))

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        image_id = self.img_ids[index]

        image_file = os.path.join(self.image_folder, image_id + '.jpg')
        image = Image.open(image_file).convert('RGB')

        anns = self._load_anns(image_id)

        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_id,
        }

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, meta)

        # mask valid TODO still necessary?
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        LOG.debug(meta)

        # log stats
        for ann in anns:
            if getattr(ann, 'iscrowd', False):
                continue
            if not np.any(ann['keypoints'][:, 2] > 0.0):
                continue
            STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})

        # transform targets
        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]
            anns_d = {}
            for ann in anns:
                for k, v in ann.items():
                    anns_d[k] = v
            anns = anns_d

        return image, anns, meta

    def _load_anns(self, image_id):
        anns = []

        human_num = 1
        while not osp.exists(osp.join(self.parsing_folder, '%s_%02d_01.png' % (image_id, human_num))):
            human_num += 1

        pose_file = osp.join(self.pose_folder, image_id + '.mat')
        pose_annotations = scipy.io.loadmat(pose_file)

        for human_id in range(1, human_num + 1):
            ann = {}

            # load parsing annotation
            name = '%s_%02d_%02d.png' % (image_id, human_num, human_id)
            parsing_file = osp.join(self.parsing_folder, name)
            parsing_annotation = cv2.imread(parsing_file)[:, :, -1]
            ann['parsing'] = parsing_annotation
            ann['segment_mask'] = np.where(parsing_annotation > 0, 1, 0)
            ann['segment_mask_original'] = np.where(parsing_annotation > 0, 1, 0)

            # load keypoint annotation
            if 'person_{}'.format(human_id-1) in pose_annotations:
                pose_annotation = pose_annotations['person_{}'.format(human_id-1)]
                keypoints = pose_annotation[:16, :]

                # In MHP: 0: visible, 1: occlusion (I guess), 2: unvisible
                keypoints[:, -1] = 2 - keypoints[:, -1]  # fit with coco keypoint visibility

                face_bbox = pose_annotation[16:18, :]  # unused
                inst_bbox = pose_annotation[18:20, :]
                inst_bbox = [inst_bbox[0, 0], inst_bbox[0, 1],
                             inst_bbox[1, 0]-inst_bbox[0, 0]+1, inst_bbox[1, 1]-inst_bbox[0, 1]+1]

                ann['bbox'] = inst_bbox
                ann['keypoints'] = keypoints.reshape(-1).tolist()
                ann['iscrowd'] = False
            else:
               continue

            anns.append(ann)

        return anns


if __name__ == '__main__':
    root = 'data/MHP'
    annfile = os.path.join(root, 'train.json')
    mhp_dataset = MHP(root, annfile, train=True)

    #mhp_dataset.__getitem__(1)
    img_ids = mhp_dataset.img_ids
    for ii in range(len(img_ids)):
        print(ii)
        mhp_dataset.__getitem__(ii)






