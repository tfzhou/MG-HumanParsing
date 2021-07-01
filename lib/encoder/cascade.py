import numpy as np
import dataclasses
import torch
import scipy.spatial
from .annrescaler import AnnRescaler


#COCO_IGNORE_INDEX = [2, 3, 4, 5]
COCO_IGNORE_INDEX = []

@dataclasses.dataclass
class Cascade:
    rescaler: AnnRescaler

    n_fields: int = 59
    offset_n_fields: int = 2
    stride: int = 4

    def __call__(self, image, anns, meta):
        return CascadeGenerator(self)(image, anns, meta)


def nearest_neightbor_search(query_points, key_points):
    mytree = scipy.spatial.cKDTree(key_points)
    _, indexes = mytree.query(query_points)
    return indexes


class CascadeGenerator(object):
    def __init__(self, config: Cascade):
        self.config = config
        self.n_fields = config.n_fields
        self.offset_n_fields = config.offset_n_fields
        self.stride = config.stride

    def __call__(self, image, anns, meta):
        h, w = image.shape[1:3]
        mask = np.zeros((h, w), dtype=np.float32)
        edge_mask = np.zeros((1, h, w), dtype=np.float32)
        flag = 0.
        for ann in anns:
            if 'parsing' in ann:
                flag = 1.
                single_mask = np.copy(ann['parsing'])
                single_mask_bool = np.where(single_mask > 0, 1, 0)
                mask[single_mask_bool > 0] = single_mask[single_mask_bool > 0]

                if 'segment_mask' in ann:
                    ignore_indexs = \
                        np.where((ann['segment_mask'] - single_mask_bool) > 0)
                    mask[ignore_indexs[0], ignore_indexs[1]] = 255  # ignore
            elif 'segment_mask' in ann:
                ignore_indexs = np.where(ann['segment_mask'] > 0)
                mask[ignore_indexs[0], ignore_indexs[1]] = 255  # ignore

            if 'edge' in ann:
                single_mask = np.copy(ann['edge'])
                single_mask[single_mask == 255] = 0
                single_mask_bool = np.where(single_mask > 0, 1, 0)
                edge_mask[0, single_mask_bool > 0] += single_mask[single_mask_bool > 0]

        edge_mask = np.where(edge_mask > 0, 1, 0)

        ### generate offset
        h, w = image.shape[1:3]
        h, w = h // self.stride + 1, w // self.stride + 1

        offset = np.zeros((5, h, w), dtype=np.float32)
        weights = np.zeros((1, h, w), dtype=np.float32)
        instance = np.zeros((1, h, w), dtype=np.float32)

        keypoints = self.config.rescaler.keypoint_sets(anns)
        segmentations = self.config.rescaler.segmentation(anns)

        num = len(segmentations)
        for iid in range(num):
            keypoint = keypoints[iid, ...]
            segmentation = segmentations[iid]
            if segmentation is None:
                continue

            segmentation[segmentation == 255] = 0
            mask_index = np.where(segmentation > 0)
            y_index, x_index = mask_index[0], mask_index[1]

            if len(y_index) == 0 or len(x_index) == 0:
                continue

            weights[0, y_index, x_index] = 1
            instance[0, y_index, x_index] = iid

            query_points = np.dstack([y_index.ravel(), x_index.ravel()])[0]

            key_points = []
            for ii, x in enumerate(keypoint):
                if x[0] > 0 and x[1] > 0 and x[2] > 0:
                    if ii+1 not in COCO_IGNORE_INDEX:
                        key_points.append([x[1], x[0]])
            if len(key_points) == 0:
                continue
            key_points = np.array(key_points)
            #key_points = np.array(
            #    [[x[1], x[0]] for x in keypoints if x[-1] > 0 and x[0] > 0 and x[1] > 0])

            indexes = nearest_neightbor_search(query_points, key_points)

            for ii in range(len(key_points)):
                points = query_points[indexes == ii]
                yy, xx = key_points[ii][0], key_points[ii][1]
                x_offset = xx - points[:, 1]
                y_offset = yy - points[:, 0]

                # offset
                offset[0, points[:, 0], points[:, 1]] = y_offset
                offset[1, points[:, 0], points[:, 1]] = x_offset

                # target
                offset[2, points[:, 0], points[:, 1]] = yy
                offset[3, points[:, 0], points[:, 1]] = xx

                # label
                offset[4, points[:, 0], points[:, 1]] = ii+1

        return {'semantic': (torch.from_numpy(mask),
                             torch.from_numpy(edge_mask), flag),
                'offset': (torch.from_numpy(offset),
                           torch.from_numpy(weights),
                           torch.from_numpy(instance))}


def create_label_colormap():
    """
    Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """

    colormap = np.zeros((256, 3), dtype=np.uint8)

    colormap[0] = [0, 0, 0]
    colormap[1] = [128, 0, 0]
    colormap[2] = [255, 0, 0]
    colormap[3] = [0, 85, 0]
    colormap[4] = [170, 0, 51]
    colormap[5] = [255, 85, 0]
    colormap[6] = [0, 0, 85]
    colormap[7] = [0, 119, 221]
    colormap[8] = [85, 85, 0]
    colormap[9] = [0, 85, 85]
    colormap[10] = [85, 51, 0]
    colormap[11] = [52, 86, 128]
    colormap[12] = [0, 128, 0]
    colormap[13] = [0, 0, 255]
    colormap[14] = [51, 170, 221]
    colormap[15] = [0, 255, 255]
    colormap[16] = [85, 255, 170]
    colormap[17] = [170, 255, 85]
    colormap[18] = [255, 255, 0]
    colormap[19] = [255, 170, 0]

    return colormap
