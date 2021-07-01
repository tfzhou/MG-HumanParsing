import numpy as np
import dataclasses
import torch


@dataclasses.dataclass
class Pdf:
    n_fields: 15

    def __call__(self, image, anns, meta):
        return PdfGenerator(self)(image, anns, meta)


class PdfGenerator(object):
    def __init__(self, config: Pdf):
        self.n_fields = config.n_fields

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
                    ignore_indexs =\
                        np.where((ann['segment_mask'] - single_mask_bool) > 0)
                    mask[ignore_indexs[0], ignore_indexs[1]] = 255  # ignore

            if 'edge' in ann:
                single_mask = np.copy(ann['edge'])
                single_mask[single_mask == 255] = 0
                single_mask_bool = np.where(single_mask > 0, 1, 0)
                edge_mask[0, single_mask_bool > 0] += single_mask[single_mask_bool > 0]

        edge_mask = np.where(edge_mask > 0, 1, 0)

        return {'semantic': (torch.from_numpy(mask),
                             torch.from_numpy(edge_mask), flag)}


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
