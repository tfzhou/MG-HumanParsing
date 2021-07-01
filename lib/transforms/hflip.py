import copy
import logging

import numpy as np
import PIL

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class _HorizontalSwapParsing(object):
    def __init__(self, categories, hflip):
        self.categories = categories
        self.hflip = hflip

    def __call__(self, mask):
        mask_flip = mask[:, ::-1]

        target = copy.deepcopy(mask_flip)

        part_ids = np.unique(mask)
        for source_i in part_ids:
            source_i = int(source_i)
            if source_i == 0 or source_i == 255:
                continue

            source_name = self.categories[source_i-1]
            target_name = self.hflip.get(source_name)

            if target_name:
                target_i = self.categories.index(target_name) + 1
            else:
                target_i = source_i

            target[mask_flip == source_i] = target_i

        return target


class _HorizontalSwap():
    def __init__(self, keypoints, hflip):
        self.keypoints = keypoints
        self.hflip = hflip

    def __call__(self, keypoints):
        target = np.zeros(keypoints.shape)

        for source_i, xyv in enumerate(keypoints):
            source_name = self.keypoints[source_i]
            target_name = self.hflip.get(source_name)
            if target_name:
                target_i = self.keypoints.index(target_name)
            else:
                target_i = source_i
            target[target_i] = xyv

        return target


class HFlip(Preprocess):
    def __init__(self, keypoints, hflip, parsing_categories=None, parsing_hflip=None):
        self.swap = _HorizontalSwap(keypoints, hflip)

        self.parsing_swap = None
        if parsing_categories is not None and parsing_hflip is not None:
            self.parsing_swap = _HorizontalSwapParsing(categories=parsing_categories,
                                                       hflip=parsing_hflip)

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, _ = image.size
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        for ann in anns:
            ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
            if self.swap is not None and not ann['iscrowd']:
                ann['keypoints'] = self.swap(ann['keypoints'])
                meta['horizontal_swap'] = self.swap
            ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w

            if 'parsing' in ann and self.parsing_swap is not None:
                ann['parsing'] = self.parsing_swap(ann['parsing'])
            if 'segment_mask' in ann:
                ann['segment_mask'] = ann['segment_mask'][:, ::-1]
            if 'edge' in ann:
                ann['edge'] = ann['edge'][:, ::-1]

        assert meta['hflip'] is False
        meta['hflip'] = True

        meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) + w

        return image, anns, meta
