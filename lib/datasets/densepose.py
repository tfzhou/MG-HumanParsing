from collections import defaultdict
import copy
import logging
import os

import numpy as np
import torch.utils.data
from PIL import Image

from .. import transforms, utils

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class DensePose(torch.utils.data.Dataset):
    """`DensePose <http://densepose.org/#dataset>`_ Dataset.

    Args:
        image_dir (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
    """

    def __init__(self, image_dir, ann_file, *,
                 coco_ann_file=None,
                 target_transforms=None,
                 n_images=None,
                 preprocess=None,
                 category_ids=None,
                 image_filter='keypoint-annotations'):
        if category_ids is None:
            category_ids = [1]
        self.category_ids = category_ids

        from pycocotools.coco import COCO  # pylint: disable=import-outside-toplevel
        self.image_dir = image_dir

        self.dp_coco = COCO(ann_file)
        self.dp_ids = self.dp_coco.getImgIds(catIds=self.category_ids)

        # use all keypoints annotations in coco dataset
        self.coco = None
        self.ids = []
        if coco_ann_file is not None:
            self.coco = COCO(coco_ann_file)

            if image_filter == 'all':
                self.ids = self.coco.getImgIds()
            elif image_filter == 'annotated':
                self.ids = self.coco.getImgIds(catIds=self.category_ids)
                self.filter_for_annotations()
            elif image_filter == 'keypoint-annotations':
                self.ids = self.coco.getImgIds(catIds=self.category_ids)
                self.filter_for_keypoint_annotations()
            else:
                raise Exception('unknown value for image_filter: {}'.format(image_filter))

            if n_images:
                self.ids = self.ids[:n_images]
            LOG.info('Images: %d', len(self.ids))

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

    def filter_for_keypoint_annotations(self):
        LOG.info('filter for keypoint annotations ...')

        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        LOG.info('... done.')

    def filter_for_annotations(self):
        """removes images that only contain crowd annotations"""
        LOG.info('filter for annotations ...')

        def has_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if ann.get('iscrowd'):
                    continue
                return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_annotation(image_id)]
        LOG.info('... done.')

    def class_aware_sample_weights(self, max_multiple=10.0):
        """Class aware sampling.

        To be used with PyTorch's WeightedRandomSampler.

        Reference: Solution for Large-Scale Hierarchical Object Detection
        Datasets with Incomplete Annotation and Data Imbalance
        Yuan Gao, Xingyuan Bu, Yang Hu, Hui Shen, Ti Bai, Xubin Li and Shilei Wen
        """
        ann_ids = self.coco.getAnnIds(imgIds=self.ids, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)

        category_image_counts = defaultdict(int)
        image_categories = defaultdict(set)
        for ann in anns:
            if ann['iscrowd']:
                continue
            image = ann['image_id']
            category = ann['category_id']
            if category in image_categories[image]:
                continue
            image_categories[image].add(category)
            category_image_counts[category] += 1

        weights = [
            sum(
                1.0 / category_image_counts[category_id]
                for category_id in image_categories[image_id]
            )
            for image_id in self.ids
        ]
        min_w = min(weights)
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))
        max_w = min_w * max_multiple
        weights = [min(w, max_w) for w in weights]
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))

        return weights

    def __getitem__(self, index):
        if self.coco is not None:
            image_id = self.ids[index]
        else:
            image_id = self.dp_ids[index]

        if image_id in self.dp_ids:
            ann_ids = self.dp_coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.dp_coco.loadAnns(ann_ids)
            anns = copy.deepcopy(anns)
            image_info = self.dp_coco.loadImgs(image_id)[0]
        else:
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            anns = copy.deepcopy(anns)
            image_info = self.coco.loadImgs(image_id)[0]

        LOG.debug(image_info)
        file_name = image_info['file_name']
        if 'COCO_train2014_' in file_name:
            file_name = file_name.replace('COCO_train2014_', '')
        with open(os.path.join(self.image_dir, file_name), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': file_name,
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

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

    def __len__(self):
        if self.coco is not None:
            return len(self.ids)
        return len(self.dp_ids)

    @staticmethod
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
