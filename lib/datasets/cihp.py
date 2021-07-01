import torch
import os
import cv2
import numpy as np
import random
from PIL import Image
from .. import transforms, utils


def make_dataset(root, lst):
    # append all index
    fid = open(lst, 'r')
    imgs, segs, segs_rev = [], [], []
    for line in fid.readlines():
        idx = line.strip().split(' ')[0]
        image_path = os.path.join(root, 'JPEGImages/' + str(idx) + '.jpg')
        seg_path = os.path.join(root, 'Segmentations/' + str(idx) + '.png')
        seg_rev_path = os.path.join(root, 'Segmentations_rev/' + str(idx) + '.png')
        imgs.append(image_path)
        segs.append(seg_path)
        segs_rev.append(seg_rev_path)
    return imgs, segs, segs_rev


# ###### val resize & crop ######
def scale_crop(img, seg, crop_size):
    oh, ow = seg.shape
    pad_h = max(0, crop_size - oh)
    pad_w = max(0, crop_size - ow)
    if pad_h > 0 or pad_w > 0:
        img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                     value=(0.0, 0.0, 0.0))
        seg_pad = cv2.copyMakeBorder(seg, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                     value=255)
    else:
        img_pad, seg_pad = img, seg

    img = np.asarray(img_pad[0: crop_size, 0: crop_size], np.float32)
    seg = np.asarray(seg_pad[0: crop_size, 0: crop_size], np.float32)

    return img, seg


class CIHP(torch.utils.data.Dataset):

    def __init__(self, root, list_path, crop_size=473, training=True,
                 preprocess=None, target_transforms=None):
        imgs, segs, segs_rev = make_dataset(root, list_path)

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.segs_rev = segs_rev
        self.crop_size = crop_size
        self.training = training

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms
        print('images: ', len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        name = self.imgs[index].split('/')[-1][:-4]
        #img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        img = Image.open(self.imgs[index]).convert('RGB')
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)

        anns = [{
            'parsing': seg
        }]

        meta = {
            'dataset_index': index,
            'image_id': index,
            'file_name': name,
        }

        # preprocess image and annotations
        img, anns, meta = self.preprocess(img, anns, meta)

        # transform targets
        if self.target_transforms is not None:
            anns = [t(img, anns, meta) for t in self.target_transforms]

        #assert img is not None, 'img is not none'
        #assert meta is not None, 'meta is not none'

        #for ann in anns:
        #    for key, value in ann.items():
        #        assert value is not None, '{} is not None'.format(key)

        return img, anns, meta


class ValidationLoader(torch.utils.data.Dataset):
    """evaluate on LIP val set"""

    def __init__(self, root, list_path, crop_size, test_transforms=None):
        fid = open(list_path, 'r')
        imgs, segs = [], []
        for line in fid.readlines():
            idx = line.strip().split(' ')[0]
            image_path = os.path.join(root, 'JPEGImages/' + str(idx) + '.jpg')
            seg_path = os.path.join(root, 'Segmentations/' + str(idx) + '.png')
            imgs.append(image_path)
            segs.append(seg_path)

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size
        self.test_transforms = test_transforms

    def __getitem__(self, index):
        # load data
        #mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)
        ori_size = img.shape

        h, w = seg.shape
        max_size = max(w, h)
        ratio = self.crop_size / max_size
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        #img = np.array(img).astype(np.float32) - mean

        img = Image.fromarray(img[:, :, ::-1])
        img = self.test_transforms(img)

        #img = img.transpose((2, 0, 1))

        #images = img.copy()
        segmentations = seg.copy()

        return img, segmentations, np.array(ori_size), name

    def __len__(self):
        return len(self.imgs)
