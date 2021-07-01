"""Evaluation on COCO data."""

import argparse
import json
import logging
import os
import sys
import time
import zipfile
import cv2
import math
import copy
from PIL import Image
import skfmm
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, slic
from skimage.segmentation import mark_boundaries

import numpy as np
import PIL
import thop
import torch
import torch.nn.functional as F

try:
    import pycocotools.coco
    from pycocotools.cocoeval import COCOeval
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

from .annotation import Annotation, AnnotationDet
from .datasets.constants import COCO_KEYPOINTS, COCO_PERSON_SKELETON, COCO_CATEGORIES
from . import datasets, decoder, network, show, transforms, visualizer, __version__
from .show.flow_vis import flow_compute_color
from .evaluation.metrics import InstanceMetrics
from .datasets.constants import DENSEPOSE_CATEGORIES


IMAGE_DIR_VAL = 'data/coco/images/train2017/'
ANNOTATIONS_VAL = 'data/coco/annotations/densepose_coco_train2017.json'

LOG = logging.getLogger(__name__)


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


palette = get_palette(256)


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


def compute_gt_confidence(gt_instance, class_map):
    confs = []
    for label in class_map.keys():
        cls = class_map[label]
        confs.append([label, cls, 1])

    return confs


def compute_confidence(semantic_scores, class_map, instance_label,
                       joint_score_map, skeleton_score_map):
    confs = []
    for label in class_map.keys():
        cls = class_map[label]
        confidence = semantic_scores[cls, :, :].reshape(-1)[
            np.where(instance_label.reshape(-1) == label)
        ]
        conf_semantic = confidence.sum() / len(confidence)

        confidence = joint_score_map.reshape(-1)[
            np.where(instance_label.reshape(-1) == label)
        ]
        conf_joint = confidence.sum() / len(confidence)

        confidence = skeleton_score_map.reshape(-1)[
            np.where(instance_label.reshape(-1) == label)
        ]
        conf_skeleton = confidence.sum() / len(confidence)

        conf_final = pow(conf_semantic * conf_joint * conf_skeleton, -3)
        confs.append([cls, conf_final])

    return confs


def get_instance(cat_gt, human_gt):
    instance_gt = np.zeros_like(cat_gt, dtype=np.uint8)

    human_ids = np.unique(human_gt)[1:]
    class_map = {}

    total_part_num = 0
    for id in human_ids:
        human_part_label = (np.where(human_gt == id, 1, 0) * cat_gt).astype(np.uint8)
        part_classes = np.unique(human_part_label)

        exceed = False
        for part_id in part_classes:
            if part_id == 0:
                continue

            total_part_num += 1

            if total_part_num > 255:
                print(
                    "total_part_num exceed, return current instance map: {}".format(
                        total_part_num)
                )
                exceed = True
                break

            class_map[total_part_num] = part_id
            instance_gt[np.where(human_part_label == part_id)] = total_part_num
        if exceed:
            break

    # Make instance id continous.
    ori_cur_labels = np.unique(instance_gt)
    total_num_label = len(ori_cur_labels)
    if instance_gt.max() + 1 != total_num_label:
        for label in range(1, total_num_label):
            instance_gt[instance_gt == ori_cur_labels[label]] = label

    final_class_map = {}
    for label in range(1, total_num_label):
        if label >= 1:
            final_class_map[label] = class_map[ori_cur_labels[label]]

    return instance_gt, final_class_map


class EvalInstance(object):
    def __init__(self, num_classes, categories):
        self.offset_vis = {}
        self.bbox_vis = {}
        self.human_vis = {}
        self.instance_vis = {}
        self.confs_vis = {}
        self.gt_instance_vis = {}
        self.gt_confs_vis = {}
        self.superpixel_vis = {}
        self.center_vis = {}
        self.center_offset_vis = {}
        self.offset_to_vis = {}

        self.image_ids = []

        self.num_classes = num_classes
        self.categories = categories

    def from_predictions(self, offset, semantic, semantic_scores, pose, edge,
                         input_size, meta, gt_semantic, gt_human,
                         pred_center=None, pred_center_offset=None):
        image_id = int(meta['image_id'])

        target_size = semantic.shape
        offset = self._restore_offset_size(offset, meta,
                                           input_size, target_size)

        if pred_center_offset is not None:
            pred_center_offset = self._restore_offset_size(pred_center_offset, meta,
                                                           input_size, target_size)

        if pred_center is not None:
            pred_center = self._restore_center_size(pred_center, meta,
                                                    input_size, target_size)

        pose = self._filter_annotations(pose)

        human, joint_score_map, skeleton_score_map, offset =\
            self._group_pixels(offset, semantic, pose, edge)

        #filename = os.path.join(IMAGE_DIR_VAL, '0'*(12-len(str(image_id))) + str(image_id) + '.jpg')
        #human = self._do_superpixel_refinement(filename, human)

        # process predictions
        instance, class_map = get_instance(semantic, human)

        confs = compute_confidence(semantic_scores, class_map, instance,
                                   joint_score_map, skeleton_score_map)
        self.confs_vis[image_id] = confs

        # process ground-truths
        gt_instance, gt_class_map = get_instance(gt_semantic, gt_human)
        gt_confs = compute_gt_confidence(gt_instance, gt_class_map)
        self.gt_confs_vis[image_id] = gt_confs

        # cache for debug
        if offset is not None:
            offset_vis = flow_compute_color(offset[:, :, 1], offset[:, :, 0])
        else:
            offset_vis = None

        ###
        y_index, x_index = np.where(semantic > 0)
        try:
            offset_vector = offset[y_index, x_index, 0:2]
            xy_offset = offset_vector + np.stack((x_index, y_index)).T
            offset_copy = offset.copy()
            offset_copy[y_index, x_index, 0] = xy_offset[:, 0]
            offset_copy[y_index, x_index, 1] = xy_offset[:, 1]
            b = np.sum(offset_copy*offset_copy, axis=-1)
            b = b/ np.max(b) * 255
            r = offset_copy[:, :, 0] / np.max(offset_copy[:, :, 0]) * 255
            g = offset_copy[:, :, 1] / np.max(offset_copy[:, :, 1]) * 255
            r = r[..., None]
            g = g[..., None]
            b = b[..., None]
            embedding = np.concatenate((r, g, b), axis=2)
            self.offset_to_vis[image_id] = embedding
        except:
            pass

        self.human_vis[image_id] = human
        self.instance_vis[image_id] = instance
        self.gt_instance_vis[image_id] = gt_instance
        self.offset_vis[image_id] = offset_vis
        self.bbox_vis[image_id] = [x['bbox'] for x in pose if x['score'] > 0.1]
        if pred_center is not None:
            self.center_vis[image_id] = pred_center * 255
        if pred_center_offset is not None:
            self.center_offset_vis[image_id] = \
                flow_compute_color(pred_center_offset[:, :, 1], pred_center_offset[:, :, 0])

    def _do_superpixel_refinement(self, filename, instance):
        im = cv2.imread(filename)[:, :, ::-1]

        if instance is None:
            return instance

        segments = slic(im, n_segments=1150)
        self.superpixel_vis[os.path.basename(filename)[:-4]] = [im, segments]
        segment_ids = np.unique(segments).tolist()

        mask = np.where(instance > 0, 1, 0)

        new_instance = copy.deepcopy(instance)
        for sid in segment_ids:
            y_index, x_index = np.where(segments == sid)
            instance_ids = instance[y_index, x_index]
            instance_ids = instance_ids.tolist()

            max_id = max(set(instance_ids), key=instance_ids.count)
            if max_id == 0:
                continue
            new_instance[y_index, x_index] = max_id
        new_instance = new_instance * mask

        return new_instance

    # compute the average geodesic distance between all pixels to each skeleton
    def _skeleton_distance(self, pose, edge):
        skeleton_distance = []
        for pid, ann in enumerate(pose):
            kps = ann['keypoints']
            kps = np.reshape(kps, (-1, 3))

            all_dt = np.ones((edge.shape[0], edge.shape[1], 17))
            for ii in range(17):
                try:
                    score = kps[ii, 2]
                    if score < 0.1:
                        continue

                    x, y = int(kps[ii][0]), int(kps[ii][1])
                    m = np.ones_like(edge).astype(float)
                    m[y, x] = 0

                    r_edge = edge > 0.2
                    m = np.ma.masked_array(m, r_edge)
                    dt = skfmm.distance(m)
                except:
                    continue

                all_dt[:, :, ii] = dt

            mean_dt = np.mean(all_dt, axis=-1)

            skeleton_distance.append(mean_dt)

            plt.figure()
            plt.imshow(mean_dt)
            for ii in range(17):
                if kps[ii][2] < 0.1:
                    continue
                plt.plot(kps[ii][0], kps[ii][1], 'r+')
            plt.savefig('{}.jpg'.format(pid))
            plt.close()

            plt.figure()
            plt.imshow(edge > 0.2)
            plt.savefig('edge.jpg')
            plt.close()

        return skeleton_distance

    def _group_pixels(self, offset, semantic, pose, edge):
        person_mask = np.where(semantic > 0, 1, 0)
        offset[:, :, 0] *= person_mask
        offset[:, :, 1] *= person_mask

        im_h, im_w = person_mask.shape
        min_size = min(im_h, im_w)

        y_index, x_index = np.where(semantic > 0)
        offset_vector = offset[y_index, x_index, 0:2]
        xy_offset = offset_vector[:, ::-1] + np.stack((x_index, y_index)).T

        xy_list, id_list, joint_score_list, person_score_list, scale_list =\
            self._prepare_sparse_instance(pose, person_mask)

        if len(xy_list) == 0:
            return None, None, None, None

        xy_list = np.concatenate(xy_list, axis=0)
        xy_pose = np.array(xy_list)

        xy_offset = xy_offset[None, ...]  # 1 * N * 2
        xy_pose = xy_pose[:, None, ...]  # K * 1 * 2

        # joint distance (local metric)
        score_list = [s1 + s2 for s1, s2 in zip(joint_score_list, person_score_list)]
        score = np.array(score_list)
        joint_distance = np.linalg.norm(xy_pose - xy_offset, axis=-1)  # K * N
        joint_distance = joint_distance.T / (score + 1e-6) / (np.array(scale_list) + 1e-6)
        distance = joint_distance.T

        if 0:
            # skeleton distance (global metric)
            skeleton_distance = self._skeleton_distance(pose, edge)
            skeleton_num = len(skeleton_distance)

            assert skeleton_num == joint_distance.shape[0] // 17

            distance = np.zeros_like(joint_distance, dtype=np.float32)
            for ii in range(skeleton_num):
                jnt_dist = joint_distance[ii*17:(ii+1)*17, :]  # 17 x N
                skl_dist = skeleton_distance[ii][y_index, x_index]  # 1 x N
                skl_dist = np.reshape(skl_dist, (1, -1))

                distance[ii*17:(ii+1)*17, :] = jnt_dist * skl_dist

        index = np.argmin(distance, axis=0)
        min_dist = np.amin(distance, axis=0)
        instance_list = np.array(id_list)
        instance_id = instance_list[index] + 1

        instance = np.zeros_like(semantic, dtype=np.uint8)
        instance[y_index, x_index] = instance_id

        # remove instances with few pixels
        instance_id_list = instance_id.tolist()
        instance_ids = list(set(instance_id_list))
        counts = [instance_id_list.count(x) for x in instance_ids]
        for id, count in zip(instance_ids, counts):
            if count < min_size / 10.:
                instance[instance == id] = 0

        joint_score_map = np.zeros_like(semantic, dtype=np.float)
        joint_score_map[y_index, x_index] = np.array(joint_score_list)[index]

        skeleton_score_map = np.zeros_like(semantic, dtype=np.float)
        skeleton_score_map[y_index, x_index] = np.array(person_score_list)[index]

        return instance, joint_score_map, skeleton_score_map, offset

    def _filter_annotations(self, annotations, th_skl=0.3, th_joint=0.1):
        annotations = [ann for ann in annotations if ann['score'] >= th_skl]

        new_annotations = []
        for ann in annotations:
            keypoints = ann['keypoints']
            keypoints = np.reshape(keypoints, (-1, 3))
            scores = keypoints[:, -1].tolist()
            size = len([s for s in scores if s > th_joint])
            if size > 2:
                new_annotations.append(ann)

        return new_annotations

    def _prepare_sparse_instance(self, annotations, fg_mask=None):
        xy_list = []
        id_list = []
        scale_list = []
        joint_score_list = []
        person_score_list = []

        if fg_mask is not None:
            fg_mask[fg_mask > 0] = 1

        for pid, ann in enumerate(annotations):
            kps = ann['keypoints']
            kps = np.reshape(kps, (-1, 3))


            bbox = ann['bbox']

            if fg_mask is not None:
                x, y, h, w = bbox[:4]
                box_size = h * w
                fg_conf = np.sum(fg_mask) / box_size

                if fg_conf < 0.1:
                    continue

            xy_list.append(kps[:, :-1])
            id_list.extend([pid] * kps.shape[0])
            scale = math.sqrt(bbox[2] * bbox[3])
            scale_list.extend([scale] * kps.shape[0])
            joint_score_list.extend(kps[:, -1].tolist())
            person_score_list.extend([ann['score']] * kps.shape[0])

        return xy_list, id_list, joint_score_list, person_score_list, scale_list

    def _restore_offset_size(self, offset, meta, input_size, target_size):
        offset = torch.tensor(offset)
        offset_rescale = F.interpolate(input=offset, size=input_size,
                                       mode='bilinear', align_corners=True)
        scale = (input_size[0] - 1) // (offset.shape[2] - 1)
        offset_rescale *= scale
        offset_rescale = offset_rescale[0].permute(1, 2, 0).cpu().numpy()

        offset = transforms.Preprocess.semantic_annotation_inverse(offset_rescale,
                                                                   target_size,
                                                                   meta)
        return offset

    def _restore_center_size(self, center, meta, input_size, target_size):
        center = torch.tensor(center)
        center_rescale = F.interpolate(input=center, size=input_size,
                                       mode='bilinear', align_corners=True)
        center_rescale = center_rescale[0].permute(1, 2, 0).cpu().numpy()

        center = transforms.Preprocess.semantic_annotation_inverse(center_rescale,
                                                                   target_size,
                                                                   meta)
        return center

    def summary(self):
        metric = InstanceMetrics(self.ins_output_dir, self.gt_ins_output_dir,
                                 num_classes=self.num_classes,
                                 categories=self.categories)

        AP_map = metric.compute_AP()
        print('Mean AP^r: {}'.format(
            np.nanmean(np.array(list(AP_map.values())))
        ))
        print('=' * 80)

    def write_predictions(self, output):
        output_dir = output + '.offset'
        os.makedirs(output_dir, exist_ok=True)

        LOG.info('\nWriting offset predictions to {}'.format(output_dir))
        for key, value in self.offset_vis.items():
            if value is None:
                continue
            filename = os.path.join(output_dir, str(key) + '.png')
            image = value.astype(np.uint8)

            #for bbox in self.bbox_vis[key]:
            #    pt1 = (int(bbox[0]), int(bbox[1]))
            #    pt2 = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))

            #    cv2.rectangle(image, pt1, pt2, (0, 0, 255))

            pred = Image.fromarray(image)

            pred.save(filename)

        output_dir = output + '.offset_to'
        os.makedirs(output_dir, exist_ok=True)

        LOG.info('\nWriting offset predictions to {}'.format(output_dir))
        for key, value in self.offset_to_vis.items():
            if value is None:
                continue
            filename = os.path.join(output_dir, str(key) + '.png')
            image = value.astype(np.uint8)

            for bbox in self.bbox_vis[key]:
                pt1 = (int(bbox[0]), int(bbox[1]))
                pt2 = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))

                cv2.rectangle(image, pt1, pt2, (0, 0, 255))

            pred = Image.fromarray(image)

            pred.save(filename)

        output_dir = output + '.center_offset'
        os.makedirs(output_dir, exist_ok=True)

        LOG.info('\nWriting center_offset predictions to {}'.format(output_dir))
        for key, value in self.center_offset_vis.items():
            if value is None:
                continue
            filename = os.path.join(output_dir, str(key) + '.png')
            image = value.astype(np.uint8)

            for bbox in self.bbox_vis[key]:
                pt1 = (int(bbox[0]), int(bbox[1]))
                pt2 = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))

                cv2.rectangle(image, pt1, pt2, (0, 0, 255))

            pred = Image.fromarray(image)

            pred.save(filename)

        output_dir = output + '.center'
        os.makedirs(output_dir, exist_ok=True)

        LOG.info('\nWriting center predictions to {}'.format(output_dir))
        for key, value in self.center_vis.items():
            if value is None:
                continue
            filename = os.path.join(output_dir, str(key) + '.png')
            image = value.astype(np.uint8)

            pred = Image.fromarray(image)

            pred.save(filename)

        output_dir = output + '.human'
        os.makedirs(output_dir, exist_ok=True)
        LOG.info('\nWriting human segmentation predictions to {}'.format(output_dir))
        for key, value in self.human_vis.items():
            if value is None:
                continue
            filename = os.path.join(output_dir, str(key) + '.png')
            image = value.astype(np.uint8)

            pred = Image.fromarray(image)
            pred.putpalette(palette)
            pred.save(filename)

        ins_output_dir = output + '.instance'
        os.makedirs(ins_output_dir, exist_ok=True)
        LOG.info('\nWriting instance parsing predictions to {}'.format(ins_output_dir))
        for key, value in self.instance_vis.items():
            if value is None:
                continue
            filename = os.path.join(ins_output_dir, str(key) + '.png')
            image = value.astype(np.uint8)

            pred = Image.fromarray(image)
            pred.putpalette(palette)
            pred.save(filename)

            # save confidence
            conf_file = open(os.path.join(ins_output_dir, str(key) + '.txt'), 'w')
            confs = self.confs_vis[key]
            for conf in confs:
                conf_file.write('{} {}\n'.format(conf[0], conf[1]))

        gt_ins_output_dir = output + '.instance.gt'
        os.makedirs(gt_ins_output_dir, exist_ok=True)
        LOG.info('\nWriting instance parsing predictions to {}'.format(gt_ins_output_dir))
        for key, value in self.gt_instance_vis.items():
            if value is None:
                continue
            filename = os.path.join(gt_ins_output_dir, str(key) + '.png')
            image = value.astype(np.uint8)

            pred = Image.fromarray(image)
            pred.putpalette(palette)
            pred.save(filename)

            # save confidence
            conf_file = open(os.path.join(gt_ins_output_dir, str(key) + '.txt'), 'w')
            confs = self.gt_confs_vis[key]
            for conf in confs:
                conf_file.write('{} {} {}\n'.format(conf[0], conf[1], conf[2]))

        output_dir = output + '.superpixel'
        os.makedirs(output_dir, exist_ok=True)
        LOG.info('\nWriting superpixel predictions to {}'.format(output_dir))
        for key, value in self.superpixel_vis.items():
            if value is None:
                continue
            filename = os.path.join(output_dir, str(key) + '.png')

            pred = mark_boundaries(value[0][:,:,::-1].astype(np.float), value[1])

            cv2.imwrite(filename, pred)

        self.ins_output_dir = ins_output_dir
        self.gt_ins_output_dir = gt_ins_output_dir


class EvalSemantic(object):
    def __init__(self, num_classes=15):
        self.image_ids = []
        self.num_classes = num_classes

        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self.palette = get_palette(num_classes)

        self.segm_preds = {}
        self.edge_preds = {}
        self.vote_preds = {}
        self.segm_gts = {}

    def from_predictions(self, pred_segm, input_size, meta,
                         gt=None, pred_edge=None, pred_vote=None):
        gt_semantic, gt_human = self.get_mask(gt)
        target_size = gt_semantic.shape
        segm_pred = None
        if gt_semantic is not None:
            image_id = int(meta['image_id'])
            self.image_ids.append(image_id)

            segm_pred_tensor = torch.from_numpy(pred_segm)
            segm_pred_tensor = F.interpolate(segm_pred_tensor, input_size,
                                             mode='bilinear')
            segm_pred_scores = segm_pred_tensor.squeeze(0).numpy()
            segm_pred = np.asarray(np.argmax(segm_pred_scores, axis=0), dtype=np.uint8)

            segm_pred = transforms.Preprocess.semantic_annotation_inverse(
                segm_pred, target_size, meta)
            self.segm_preds[image_id] = segm_pred

            if pred_vote is not None:
                vote_pred_tensor = torch.from_numpy(pred_vote)
                vote_pred_tensor = F.interpolate(vote_pred_tensor, input_size, mode='bilinear')
                vote_pred = vote_pred_tensor.squeeze(0).squeeze(0).numpy()
                vote_pred = transforms.Preprocess.semantic_annotation_inverse(
                    vote_pred, target_size, meta
                )
                self.vote_preds[image_id] = vote_pred

            if pred_edge is not None:
                edge_pred_tensor = torch.from_numpy(pred_edge)
                edge_pred_tensor = F.interpolate(edge_pred_tensor, input_size, mode='bilinear')
                edge_pred = edge_pred_tensor.squeeze(0).squeeze(0).numpy()
                edge_pred = transforms.Preprocess.semantic_annotation_inverse(
                    edge_pred, target_size, meta
                )
                self.edge_preds[image_id] = edge_pred

            ignore_index = gt_semantic != 255
            seg_gt_ = gt_semantic[ignore_index]
            seg_pred_ = segm_pred[ignore_index]

            self.confusion_matrix += get_confusion_matrix(seg_gt_, seg_pred_,
                                                          self.num_classes)
        if pred_edge is not None:
            return segm_pred, segm_pred_scores, gt_semantic, gt_human, edge_pred
        return segm_pred, segm_pred_scores, gt_semantic, gt_human, None

    def summary(self):
        pos = self.confusion_matrix.sum(1)
        res = self.confusion_matrix.sum(0)
        tp = np.diag(self.confusion_matrix)

        pixel_accuracy = tp.sum() / pos.sum()
        mean_accuracy = (tp / np.maximum(1.0, pos)).mean()
        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        LOG.info('\nMean IoU: {}, PixelAcc: {}, Mean Acc: {}'.format(mean_IU, pixel_accuracy, mean_accuracy))

    def write_predictions(self, output):
        output_dir = output + '.global-parsing'
        os.makedirs(output_dir, exist_ok=True)

        LOG.info('\nWriting semantic parsing predictions to {}'.format(output_dir))
        for key, value in self.segm_preds.items():
            filename = os.path.join(output_dir, str(key) + '.png')

            pred = Image.fromarray(value)
            pred.putpalette(self.palette)
            pred.save(filename)

        output_dir = output + '.global-parsing.gt'
        os.makedirs(output_dir, exist_ok=True)
        LOG.info('\nWriting semantic parsing ground-truths to {}'.format(output_dir))
        for key, value in self.segm_gts.items():
            filename = os.path.join(output_dir, str(key) + '.png')

            pred = Image.fromarray(value)
            pred.putpalette(self.palette)
            pred.save(filename)

        output_dir = output + '.edge'
        os.makedirs(output_dir, exist_ok=True)
        LOG.info('\nWriting semantic parsing predictions to {}'.format(output_dir))
        for key, value in self.edge_preds.items():
            filename = os.path.join(output_dir, str(key) + '.jpg')

            pred = Image.fromarray(value * 255).convert('L')
            pred.save(filename)

        output_dir = output + '.vote'
        os.makedirs(output_dir, exist_ok=True)

        LOG.info('\nWriting voting predictions to {}'.format(output_dir))
        for key, value in self.vote_preds.items():
            filename = os.path.join(output_dir, str(key) + '.jpg')

            pred = Image.fromarray(value * 255).convert('L')
            pred.save(filename)

    def get_mask(self, anns):
        gt_mask = None
        gt_human = None
        for ii, ann in enumerate(anns):
            if 'parsing_original' in ann:
                single_mask = np.copy(ann['parsing_original'])
                if gt_mask is None:
                    gt_mask = np.zeros_like(single_mask, dtype=np.uint8)
                if gt_human is None:
                    gt_human = np.zeros_like(single_mask, dtype=np.uint8)

                single_mask_bool = np.where(single_mask > 0, 1, 0)
                gt_mask[single_mask_bool > 0] = single_mask[single_mask_bool > 0]
                gt_human[single_mask_bool > 0] = ii + 1

        return gt_mask, gt_human


class EvalCoco(object):
    def __init__(self, coco, processor, *,
                 max_per_image=20,
                 category_ids=None,
                 iou_type='keypoints',
                 small_threshold=0.0):
        if category_ids is None:
            category_ids = [1]

        self.coco = coco
        self.processor = processor
        self.max_per_image = max_per_image
        self.category_ids = category_ids
        self.iou_type = iou_type
        self.small_threshold = small_threshold

        self.predictions = []
        self.image_ids = []
        self.eval = None
        self.decoder_time = 0.0
        self.nn_time = 0.0

        self.pose_vis = {}

        LOG.debug('max = %d, category ids = %s, iou_type = %s',
                  self.max_per_image, self.category_ids, self.iou_type)

    def stats(self, predictions=None, image_ids=None):
        # from pycocotools.cocoeval import COCOeval
        if predictions is None:
            predictions = self.predictions
        if image_ids is None:
            image_ids = self.image_ids

        coco_eval = self.coco.loadRes(predictions)

        self.eval = COCOeval(self.coco, coco_eval, iouType=self.iou_type)
        LOG.info('cat_ids: %s', self.category_ids)
        if self.category_ids:
            self.eval.params.catIds = self.category_ids

        if image_ids is not None:
            print('image ids', image_ids)
            self.eval.params.imgIds = image_ids
        self.eval.evaluate()
        self.eval.accumulate()
        self.eval.summarize()
        return self.eval.stats

    @staticmethod
    def count_ops(model, height=641, width=641):
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, height, width, device=device)
        gmacs, params = thop.profile(model, inputs=(dummy_input, ))
        LOG.info('GMACs = {0:.2f}, million params = {1:.2f}'.format(gmacs / 1e9, params / 1e6))
        return gmacs, params

    @staticmethod
    def view_annotations(meta, predictions, ground_truth):
        annotation_painter = show.AnnotationPainter()
        with open(os.path.join(IMAGE_DIR_VAL, meta['file_name']), 'rb') as f:
            cpu_image = PIL.Image.open(f).convert('RGB')

        with show.image_canvas(cpu_image) as ax:
            annotation_painter.annotations(ax, predictions)

        if ground_truth:
            with show.image_canvas(cpu_image) as ax:
                show.white_screen(ax)
                annotation_painter.annotations(ax, ground_truth, color='grey')
                annotation_painter.annotations(ax, predictions)

    def from_predictions(self, predictions, meta, debug=False, gt=None):
        image_id = int(meta['image_id'])
        self.image_ids.append(image_id)

        predictions = transforms.Preprocess.annotations_inverse(predictions, meta)
        if self.small_threshold:
            predictions = [pred for pred in predictions
                           if pred.scale(v_th=0.01) >= self.small_threshold]
        if len(predictions) > self.max_per_image:
            predictions = predictions[:self.max_per_image]

        self.pose_vis[meta['file_name']] = predictions

        if debug:
            gt_anns = []
            for g in gt:
                if 'bbox' in g:
                    gt_anns.append(
                        AnnotationDet(COCO_CATEGORIES).set(g['category_id'] - 1, None, g['bbox'])
                    )
                if 'keypoints' in g:
                    gt_anns.append(
                        Annotation(COCO_KEYPOINTS, COCO_PERSON_SKELETON)
                        .set(g['keypoints'], fixed_score=None)
                    )
            gt_anns = transforms.Preprocess.annotations_inverse(gt_anns, meta)
            self.view_annotations(meta, predictions, gt_anns)

        image_annotations = []
        for pred in predictions:
            pred_data = pred.json_data()
            pred_data['image_id'] = image_id
            pred_data = {
                k: v for k, v in pred_data.items()
                if k in ('category_id', 'score', 'keypoints', 'bbox', 'image_id')
            }
            image_annotations.append(pred_data)

        # force at least one annotation per image (for pycocotools)
        if not image_annotations:
            image_annotations.append({
                'image_id': image_id,
                'category_id': 1,
                'keypoints': np.zeros((17*3,)).tolist(),
                'bbox': [0, 0, 1, 1],
                'score': 0.001,
            })

        if debug:
            self.stats(image_annotations, [image_id])
            LOG.debug(meta)

        self.predictions += image_annotations
        return image_annotations

    def write_predictions(self, filename):
        predictions = [
            {k: v for k, v in annotation.items()
             if k in ('image_id', 'category_id', 'keypoints', 'score')}
            for annotation in self.predictions
        ]
        with open(filename + '.pred.json', 'w') as f:
            json.dump(predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)

        # debug
        # visualizers
        keypoint_painter = show.KeypointPainter(
            color_connections=True,
            linewidth=6,
        )
        annotation_painter = show.AnnotationPainter(
            keypoint_painter=keypoint_painter)

        output_dir = filename + '.pose'
        os.makedirs(output_dir, exist_ok=True)
        LOG.info('\nWriting pose predictions to {}'.format(output_dir))
        for key, value in self.pose_vis.items():
            if value is None:
                continue

            imagefile = os.path.join(IMAGE_DIR_VAL, key)
            with open(imagefile, 'rb') as f:
                cpu_image = PIL.Image.open(f).convert('RGB')

            image_out_name = os.path.join(output_dir, str(key) + '.jpg')
            LOG.debug('image output = %s', image_out_name)
            with show.image_canvas(cpu_image,
                                   image_out_name,
                                   show=False,
                                   fig_width=10.0,
                                   dpi_factor=1.0) as ax:
                annotation_painter.annotations(ax, value)


def default_output_name(args):
    output = '{}.evalcoco-{}edge{}'.format(
        args.checkpoint,
        '{}-'.format(args.dataset) if args.dataset != 'val' else '',
        args.long_edge,
    )
    if args.n:
        output += '-samples{}'.format(args.n)
    if not args.force_complete_pose:
        output += '-noforcecompletepose'
    if args.orientation_invariant or args.extended_scale:
        output += '-'
        if args.orientation_invariant:
            output += 'o'
        if args.extended_scale:
            output += 's'
    if args.two_scale:
        output += '-twoscale'
    if args.multi_scale:
        output += '-multiscale'
        if args.multi_scale_hflip:
            output += 'whflip'

    return output


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

    parser.add_argument('--use-superpixel', default=False, type=bool)
    parser.add_argument('--eval-pose', default=True, type=bool)
    parser.add_argument('--eval-semantic', default=True, type=bool)
    parser.add_argument('--eval-instance', default=True, type=bool)

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

    if args.dataset == 'val' and not args.detection_annotations:
        args.image_dir = IMAGE_DIR_VAL
        args.annotation_file = ANNOTATIONS_VAL
    else:
        raise Exception

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

    # generate a default output filename
    if args.output is None:
        args.output = default_output_name(args)

    return args


def write_evaluations(eval_coco, filename, args, total_time, count_ops, file_size):
    if args.write_predictions:
        eval_coco.write_predictions(filename)

    n_images = len(eval_coco.image_ids)

    if args.dataset not in ('test', 'test-dev'):
        stats = eval_coco.stats()
        np.savetxt(filename + '.txt', stats)
        with open(filename + '.stats.json', 'w') as f:
            json.dump({
                'stats': stats.tolist(),
                'n_images': n_images,
                'decoder_time': eval_coco.decoder_time,
                'nn_time': eval_coco.nn_time,
                'total_time': total_time,
                'checkpoint': args.checkpoint,
                'count_ops': count_ops,
                'file_size': file_size,
            }, f)
    else:
        print('given dataset does not have ground truth, so no stats summary')

    print('n images = {}'.format(n_images))
    print('decoder time = {:.1f}s ({:.0f}ms / image)'
          ''.format(eval_coco.decoder_time, 1000 * eval_coco.decoder_time / n_images))
    print('nn time = {:.1f}s ({:.0f}ms / image)'
          ''.format(eval_coco.nn_time, 1000 * eval_coco.nn_time / n_images))
    print('total time = {:.1f}s ({:.0f}ms / image)'
          ''.format(total_time, 1000 * total_time / n_images))


def preprocess_factory(
        long_edge,
        *,
        tight_padding=False,
        extended_scale=False,
        orientation_invariant=False,
):
    preprocess = [transforms.NormalizeAnnotations()]

    if extended_scale:
        assert long_edge
        preprocess += [
            transforms.DeterministicEqualChoice([
                transforms.RescaleAbsolute(long_edge),
                transforms.RescaleAbsolute((long_edge - 1) // 2 + 1),
            ], salt=1)
        ]
    elif long_edge:
        preprocess += [transforms.RescaleAbsolute(long_edge)]

    if tight_padding:
        preprocess += [transforms.CenterPadTight(16)]
    else:
        assert long_edge
        preprocess += [transforms.CenterPad(long_edge)]

    if orientation_invariant:
        preprocess += [
            transforms.DeterministicEqualChoice([
                None,
                transforms.RotateBy90(fixed_angle=90),
                transforms.RotateBy90(fixed_angle=180),
                transforms.RotateBy90(fixed_angle=270),
            ], salt=3)
        ]

    preprocess += [transforms.EVAL_TRANSFORM]
    return transforms.Compose(preprocess)


def dataloader_from_args(args):
    preprocess = preprocess_factory(
        args.long_edge,
        tight_padding=args.batch_size == 1 and not args.multi_scale,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
    )
    data = datasets.DensePose(
        image_dir=args.image_dir,
        ann_file=args.annotation_file,
        preprocess=preprocess,
        image_filter='all' if args.all_images else 'annotated',
        category_ids=[] if args.detection_annotations else [1],
    )
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, pin_memory=args.pin_memory,
        num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

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
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.base_net = model_cpu.base_net
        model.pose_head_nets = model_cpu.pose_head_nets
        model.segm_head_nets = model_cpu.segm_head_nets
        model.head_nets = model_cpu.head_nets

    processor = decoder.factory_from_args(args, model)
    coco = pycocotools.coco.COCO(args.annotation_file)
    eval_coco = EvalCoco(
        coco,
        processor,
        max_per_image=100 if args.detection_annotations else 20,
        category_ids=[] if args.detection_annotations else [1],
        iou_type='bbox' if args.detection_annotations else 'keypoints',
    )
    eval_segm = EvalSemantic()
    eval_inst = EvalInstance(num_classes=15, categories=DENSEPOSE_CATEGORIES)

    total_start = time.time()
    loop_start = time.time()
    for batch_i, (image_tensors, anns_batch, meta_batch) in enumerate(data_loader):
        LOG.info('batch %d, last loop: %.3fs, batches per second=%.1f',
                 batch_i, time.time() - loop_start,
                 batch_i / max(1, (time.time() - total_start)))
        if batch_i < args.skip_n:
            continue
        if args.n and batch_i >= args.n:
            break

        if meta_batch[0]['file_name'] != '000000219294.jpg':
            continue

        print(meta_batch[0]['file_name'])

        #if meta_batch[0]['image_id'] != 32901:
        #    continue

        im_h, im_w = image_tensors.shape[2], image_tensors.shape[3]

        loop_start = time.time()

        if len([a
                for anns in anns_batch
                for a in anns
                if np.any(a['keypoints'][:, 2] > 0)]) < args.min_ann:
            continue

        pred_batch = processor.batch(model, image_tensors, device=args.device)
        eval_coco.decoder_time += processor.last_decoder_time
        eval_coco.nn_time += processor.last_nn_time

        pred_center, pred_center_offset = None, None
        pred_pose, pred_segm, pred_offset, pred_edge, pred_vote = \
            None, None, None, None, None
        if isinstance(pred_batch, dict):
            if 'semantic' in pred_batch:
                pred_segm = pred_batch['semantic']
            if 'offset' in pred_batch:
                pred_offset = pred_batch['offset']
            if 'pose' in pred_batch:
                pred_pose = pred_batch['pose']
            if 'edge' in pred_batch:
                pred_edge = pred_batch['edge']
            if 'vote' in pred_batch:
                pred_vote = pred_batch['vote'][:, :1, :, :]
            if 'center' in pred_batch:
                pred_center = pred_batch['center']
                pred_center_offset = pred_batch['center_offset']

        assert len(image_tensors) == len(anns_batch)
        assert len(image_tensors) == len(meta_batch)
        assert len(image_tensors) == 1, 'only support batch size = 1'

        # process pose predictions
        if pred_pose is not None and args.eval_pose is True:
            pred_pose = eval_coco.from_predictions(pred_pose[0], meta_batch[0],
                                                   debug=args.debug, gt=anns_batch[0])

        # process semantic segmentation predictions
        if pred_segm is not None and args.eval_semantic is True:
            pred_segm, pred_segm_scores, gt_semantic, gt_human, pred_edge =\
                eval_segm.from_predictions(pred_segm, (im_h, im_w),
                                           pred_edge=pred_edge,
                                           meta=meta_batch[0],
                                           gt=anns_batch[0],
                                           pred_vote=pred_vote)

        # process offset predictions
        if pred_offset is not None and args.eval_instance is True:
            eval_inst.from_predictions(pred_offset, pred_segm,
                                       pred_segm_scores,
                                       pred_pose, pred_edge,
                                       pred_center=pred_center,
                                       pred_center_offset=pred_center_offset,
                                       input_size=(im_h, im_w),
                                       meta=meta_batch[0],
                                       gt_semantic=gt_semantic,
                                       gt_human=gt_human)

        if meta_batch[0]['file_name'] != '000000219294.jpg':
            break
    total_time = time.time() - total_start

    # model stats
    count_ops = list(eval_coco.count_ops(model_cpu))
    local_checkpoint = network.local_checkpoint_path(args.checkpoint)
    file_size = os.path.getsize(local_checkpoint) if local_checkpoint else -1.0

    # write coco

    # write
    if args.eval_pose is True:
        write_evaluations(eval_coco, args.output, args, total_time, count_ops,
                          file_size)
        eval_coco.write_predictions(args.output)

    # write semantic parsing
    if args.eval_semantic is True:
        eval_segm.summary()
        eval_segm.write_predictions(args.output)

    # write instance parsing
    if args.eval_instance is True:
        eval_inst.write_predictions(args.output)
        eval_inst.summary()


if __name__ == '__main__':
    main()
