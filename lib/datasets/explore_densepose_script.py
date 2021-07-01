import os
from pycocotools.coco import COCO

COCO_IMAGE_DIR_TRAIN = 'data/coco/images/train2017/'
DENSEPOSE_ANNOTATIONS_TRAIN = 'data/coco/annotations/densepose_coco_train2017.json'
COCOKP_ANNOTATIONS_TRAIN = 'data/coco/annotations/person_keypoints_train2017.json'

dp_coco = COCO(DENSEPOSE_ANNOTATIONS_TRAIN)
dp_ids = dp_coco.getImgIds(catIds=[1])

cnt = 0
for image_id in dp_ids:
    ann_ids = dp_coco.getAnnIds(imgIds=image_id, catIds=[1])
    anns = dp_coco.loadAnns(ann_ids)

    for ann in anns:
        if 'dp_masks' in ann:
            cnt += 1
print('number of annotations: ', cnt)