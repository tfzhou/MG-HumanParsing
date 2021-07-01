import cv2
import copy
import logging

import numpy as np
import torch

from .preprocess import Preprocess
import pycocotools.mask as maskUtils
from .utils import generate_edge

LOG = logging.getLogger(__name__)


# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L413
def coco_polygon_to_mask(segm, h, w):
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        rle = segm

    m = maskUtils.decode(rle)

    return m


def dp_mask_to_mask(polys):
    semantic_mask = np.zeros((256, 256), dtype=np.uint8)
    for i in range(1, 15):
        if polys[i-1]:
            current_mask = maskUtils.decode(polys[i - 1])
            semantic_mask[current_mask > 0] = i

    return semantic_mask


class NormalizeAnnotations(Preprocess):
    @staticmethod
    def normalize_annotations(anns, w, h):
        anns = copy.deepcopy(anns)

        for ann in anns:
            if 'keypoints' not in ann:
                ann['keypoints'] = []
            if 'bbox' not in ann:
                ann['bbox'] = []

            ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)

            ann['bbox'] = np.asarray(ann['bbox'], dtype=np.float32)
            if 'bbox_original' not in ann:
                ann['bbox_original'] = np.copy(ann['bbox'])

            if 'dp_masks' in ann and 'parsing' not in ann:
                semantic_mask = np.zeros((h, w), dtype=np.uint8)
                mask = dp_mask_to_mask(ann['dp_masks'])
                bbr = np.array(ann['bbox']).astype(int)
                x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
                x2, y2 = min(x2, w), min(y2, h)

                if x1 < x2 and y1 < y2:
                    mask = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)),
                                      interpolation=cv2.INTER_NEAREST)
                    mask_bool = np.where(mask > 0, 1, 0)
                    semantic_mask[y1:y2, x1:x2][mask_bool > 0] = mask[mask_bool > 0]
                ann['parsing'] = np.copy(semantic_mask)
                ann['parsing_original'] = np.copy(semantic_mask)
            elif 'parsing' in ann and 'parsing_original' not in ann:
                ann['parsing_original'] = ann['parsing']

            if 'segmentation' in ann and 'segment_mask' not in ann:
                mask = coco_polygon_to_mask(ann['segmentation'], h, w)
                ann['segment_mask'] = np.copy(mask)
                ann['segment_mask_original'] = np.copy(mask)

            if 'edge' not in ann or 'edge_original' not in ann:
                edge = generate_edge(ann['segment_mask'])
                ann['edge'] = np.copy(edge)
                ann['edge_original'] = np.copy(edge)
        return anns

    def __call__(self, image, anns, meta):
        w, h = image.size
        anns = self.normalize_annotations(anns, w, h)

        if meta is None:
            meta = {}

        # fill meta with defaults if not already present
        w, h = image.size
        meta_from_image = {
            'offset': np.array((0.0, 0.0)),
            'scale': np.array((1.0, 1.0)),
            'rotation': {'angle': 0.0, 'width': None, 'height': None},
            'valid_area': np.array((0.0, 0.0, w - 1, h - 1)),
            'hflip': False,
            'width_height': np.array((w, h)),
        }
        for k, v in meta_from_image.items():
            if k not in meta:
                meta[k] = v

        return image, anns, meta


class AnnotationJitter(Preprocess):
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        for ann in anns:
            keypoints_xy = ann['keypoints'][:, :2]
            sym_rnd_kp = (torch.rand(*keypoints_xy.shape).numpy() - 0.5) * 2.0
            keypoints_xy += self.epsilon * sym_rnd_kp

            sym_rnd_bbox = (torch.rand((4,)).numpy() - 0.5) * 2.0
            ann['bbox'] += 0.5 * self.epsilon * sym_rnd_bbox

        return image, anns, meta
