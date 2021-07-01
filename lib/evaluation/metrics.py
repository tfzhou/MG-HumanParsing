import os
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm




class GlobalMetrics():
    """
  """

    def __init__(self, global_result_paths, global_gt_paths, num_classes):
        """
    """
        self.global_result_paths = global_result_paths
        self.global_gt_paths = global_gt_paths
        self.num_classes = num_classes
        self._compute_hist()

    def get_pixel_accuray(self):
        """
    """
        return self.num_correct_pix.sum() / self.hist.sum()

    def get_mean_pixel_accuracy(self):
        """
    """
        # Pixel accuracy for each class.
        pixel_accuracys = self.num_correct_pix / self.num_gt_pix
        return np.nanmean(pixel_accuracys)

    def get_mean_IoU(self):
        """
    """
        union = self.num_gt_pix + self.hist.sum(0) - self.num_correct_pix
        # IoU for each class.
        IoUs = self.num_correct_pix / union
        return np.nanmean(IoUs)

    def get_frequency_weighted_IoU(self):
        """
    """
        freq = self.num_gt_pix / self.hist.sum()
        union = self.num_gt_pix + self.hist.sum(0) - self.num_correct_pix
        IoUs = self.num_correct_pix / union

        return (freq[freq > 0] * IoUs[freq > 0]).sum()

    def _compute_hist(self):
        """
    """
        # Number of classes(include background).
        num_classes = self.num_classes

        # hist[i, j] means the number of pixels that are predicted to class j,
        # whose actual label are class i.
        hist = np.zeros((num_classes, num_classes))
        for res_path, gt_path in zip(self.global_result_paths,
                                     self.global_gt_paths):
            gt = PILImage.open(gt_path)
            gt = np.array(gt, dtype=np.int32)
            res = PILImage.open(res_path)
            res = np.array(res, dtype=np.int32)

            gt_size = gt.shape
            res_size = res.shape
            assert (gt_size == res_size)

            hist += self._fast_hist(gt, res, num_classes)

        self.hist = hist
        # Number of correctly classified pixels.
        self.num_correct_pix = np.diag(self.hist)
        # Number of pixels of each classes in groundtruth.
        self.num_gt_pix = self.hist.sum(1)

    def _fast_hist(self, a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(
            n * a[k].astype(int) + b[k],
            minlength=n ** 2
        ).reshape(n, n)


class InstanceMetrics():
    def __init__(self, instance_pred_folder, instance_gt_folder, num_classes, categories):
        self.INSTANCE_PRED_FOLDER = instance_pred_folder
        self.INSTANCE_GT_FOLDER = instance_gt_folder
        self.NUM_CLASSES = num_classes
        self.IoU_TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.categories = categories

    def compute_AP(self):
        image_name_list = [
            x[:-4] for x in os.listdir(self.INSTANCE_PRED_FOLDER) if x[-3:] == 'txt'
        ]

        # For every class(except for background, 0),
        # compute APs under different IoU threshold.
        AP = np.zeros((self.NUM_CLASSES - 1, len(self.IoU_TH)))
        with tqdm(total=self.NUM_CLASSES - 1) as pbar:
            pbar.set_description('Computing AP^r')
            for class_id in range(1, self.NUM_CLASSES):
                AP[class_id - 1, :] = self._compute_class_ap(image_name_list, class_id)
                pbar.update(1)

        mAP_per_class = np.mean(AP, axis=1)
        for i, cat in enumerate(self.categories):
            print(cat, ': ', mAP_per_class[i])
            #print(AP[i, :])

        # AP under each threshold
        mAP = np.nanmean(AP, axis=0)

        AP_map = {}
        for i, thre in enumerate(self.IoU_TH):
            AP_map[thre] = mAP[i]
        # print('mAP: {}, {}'.format(mAP, np.nanmean(mAP)))

        return AP_map

    def _compute_class_ap(self, image_name_list, class_id):
        """
    """
        num_IoU_TH = len(self.IoU_TH)
        AP = np.zeros((num_IoU_TH))

        num_gt_masks = 0
        num_pred_masks = 0
        true_pos = []
        false_pos = []
        # [TODO] What's this mean?
        scores = []

        for i in range(num_IoU_TH):
            true_pos.append([])
            false_pos.append([])

        for image_name in image_name_list:
            instance_img_gt = PILImage.open(
                os.path.join(self.INSTANCE_GT_FOLDER, image_name + '.png')
            )
            instance_img_gt = np.array(instance_img_gt)

            # File for accelerating computation.
            # Each line has three numbers: "instance_part_id class_id human_id".
            rfp = open(
                os.path.join(self.INSTANCE_GT_FOLDER, image_name + '.txt'),
                'r'
            )
            # Instance ID from groundtruth file.
            gt_part_id = []
            gt_id = []
            for line in rfp.readlines():
                line = line.strip().split(' ')
                gt_part_id.append([int(line[0]), int(line[1])])
                if int(line[1]) == class_id:
                    gt_id.append(int(line[0]))
            rfp.close()

            instance_img_pred = PILImage.open(
                os.path.join(self.INSTANCE_PRED_FOLDER, image_name + '.png')
            )
            instance_img_pred = np.array(instance_img_pred)
            # Each line has two numbers: "class_id score"
            rfp = open(
                os.path.join(self.INSTANCE_PRED_FOLDER, image_name + '.txt'),
                'r'
            )
            # Instance ID from predicted file.
            pred_id = []
            pred_scores = []
            for i, line in enumerate(rfp.readlines()):
                line = line.strip().split(' ')
                if int(line[0]) == class_id:
                    pred_id.append(i + 1)
                    pred_scores.append(float(line[1]))
            rfp.close()

            # Mask for specified class, i.e., *class_id*
            gt_masks, num_gt_instance = self._split_masks(instance_img_gt,
                                                          set(gt_id))
            pred_masks, num_pred_instance = self._split_masks(instance_img_pred,
                                                              set(pred_id))

            num_gt_masks += num_gt_instance
            num_pred_masks += num_pred_instance

            if num_pred_instance == 0:
                continue

            # Collect scores from all the test images that
            # belong to class *class_id*
            scores += pred_scores

            if num_gt_instance == 0:
                for i in range(num_pred_instance):
                    for k in range(num_IoU_TH):
                        false_pos[k].append(1)
                        true_pos[k].append(0)
                continue

            gt_masks = np.stack(gt_masks)
            pred_masks = np.stack(pred_masks)
            # Compute IoU overlaps [pred_masks, gt_makss]
            # overlaps[i, j]: IoU between predicted mask i and gt mask j
            overlaps = self._compute_mask_overlaps(pred_masks, gt_masks)

            max_overlap_index = np.argmax(overlaps, axis=1)

            for i in np.arange(len(max_overlap_index)):
                max_IoU = overlaps[i][max_overlap_index[i]]
                for k in range(num_IoU_TH):
                    if max_IoU > self.IoU_TH[k]:
                        true_pos[k].append(1)
                        false_pos[k].append(0)
                    else:
                        true_pos[k].append(0)
                        false_pos[k].append(1)

        ind = np.argsort(scores)[::-1]

        for k in range(num_IoU_TH):
            m_tp = np.array(true_pos[k])[ind]
            m_fp = np.array(false_pos[k])[ind]

            m_tp = np.cumsum(m_tp)
            m_fp = np.cumsum(m_fp)
            recall = m_tp / float(num_gt_masks)
            precision = m_tp / np.maximum(m_fp + m_tp, np.finfo(np.float64).eps)

            # Compute mean AP over recall range
            AP[k] = self._voc_ap(recall, precision, False)

        return AP

    def _voc_ap(self, recall, precision, use_07_metric=False):
        """
    Compute VOC AP given precision and recall. If use_07_metric is true,
    uses the VOC 07 11 point method (default:False).
    """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            # arange([start, ]stop, [step, ]dtype=None)
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], recall, [1.]))
            mpre = np.concatenate(([0.], precision, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def _compute_mask_overlaps(self, pred_masks, gt_masks):
        """
    Computes IoU overlaps between two sets of masks.
    For better performance, pass the largest set first and the smaller second.

    Input:
      pred_masks --  [num_instances, h, width], Instance masks
      gt_masks   --  [num_instances, h, width], ground truth
    """
        pred_areas = self._count_nonzero(pred_masks)
        gt_areas = self._count_nonzero(gt_masks)

        overlaps = np.zeros((pred_masks.shape[0], gt_masks.shape[0]))
        for i in range(overlaps.shape[1]):
            gt_mask = gt_masks[i]
            overlaps[:, i] = self._compute_mask_IoU(gt_mask, pred_masks,
                                                    gt_areas[i], pred_areas)

        return overlaps

    def _compute_mask_IoU(self, gt_mask, pred_masks,
                          gt_mask_area, pred_mask_areas):
        """
    Calculates IoU of the specific groundtruth mask
    with the array of all the predicted mask.

    Input:
      gt_mask         -- A mask of groundtruth with shape of [h, w].
      pred_masks      -- An array represents a set of masks,
                         with shape [num_instances, h, w].
      gt_mask_area    -- An integer represents the area of gt_mask.
      pred_mask_areas -- A set of integers represents areas of pred_masks.
    """

        # logical_and() can be broadcasting.
        intersection = np.logical_and(gt_mask, pred_masks)
        # True then the corresponding position of output is 1, otherwise is 0.
        intersection = np.where(intersection == True, 1, 0).astype(np.uint8)  # noqa
        intersection = self._count_nonzero(intersection)

        mask_gt_areas = np.full(len(pred_mask_areas), gt_mask_area)

        union = mask_gt_areas + pred_mask_areas[:] - intersection[:]

        iou = intersection / union

        return iou

    def _split_masks(self, instance_img, id_to_convert=None):
        """
    Split a single mixed mask into several class-specified masks.

    Input:
      instance_img  -- An index map with shape [h, w]
      id_to_convert -- A list of instance part IDs that suppose to
                       extract from instance_img, if *None*, extract all the
                       ID maps except for background.

    Return:
      masks -- A collection of masks with shape [num_instance, h, w]
    """
        masks = []

        instance_ids = np.unique(instance_img)
        background_id_index = np.where(instance_ids == 0)[0]
        instance_ids = np.delete(instance_ids, background_id_index)

        if id_to_convert is None:
            for i in instance_ids:
                masks.append((instance_img == i).astype(np.uint8))
        else:
            for i in instance_ids:
                if i in id_to_convert:
                    masks.append((instance_img == i).astype(np.uint8))

        return masks, len(masks)

    def _count_nonzero(self, masks):
        """
    Compute the total number of nonzero items in each mask.

    Input:
      masks -- a three-dimension array with shape [num_instance, h, w],
               includes *num_instance* of two-dimension mask arrays.

    Return:
      nonzero_count -- A tuple with *num_instance* digital elements,
                       each of which represents the area of specific instance.
    """
        area = []
        for i in masks:
            _, a = np.nonzero(i)
            area.append(a.shape[0])
        area = tuple(area)
        return area
