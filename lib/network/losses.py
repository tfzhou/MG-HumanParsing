"""Losses."""

import logging
import torch
import torch.nn.functional as F

from . import heads
from ..show.flow_vis import flow_compute_color
from PIL import Image
from datetime import datetime

from .lovasz_loss import lovasz_softmax_flat, flatten_probas, lovasz_hinge

LOG = logging.getLogger(__name__)


class CenterLoss(torch.nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.field_names = ['center', 'center_offset']

        self.offset_loss = torch.nn.L1Loss(reduction='none')
        self.center_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, preds, targets):
        pred_center, pred_offset = preds['center'], preds['center_offset']
        target_center, target_offset = targets[0], targets[1]
        weight = targets[2]

        offset_loss_weights = weight[:, None, :, :].expand_as(target_offset)
        center_loss_weights = weight[:, None, :, :].expand_as(target_center)

        pred_h, pred_w = pred_offset.shape[2], pred_offset.shape[3]
        target_h, target_w = target_center.shape[2], target_center.shape[3]
        pred_center = F.interpolate(pred_center, size=(target_h, target_w),
                                    mode='bilinear', align_corners=True)
        pred_offset = F.interpolate(pred_offset, size=(target_h, target_w),
                                    mode='bilinear', align_corners=True)
        scale = (target_h - 1) // (pred_h - 1)
        pred_offset *= scale

        offset_loss = self.offset_loss(pred_offset, target_offset) * offset_loss_weights
        if offset_loss_weights.sum() > 0:
            offset_loss = offset_loss.sum() / offset_loss_weights.sum() / 10
        else:
            offset_loss = offset_loss.sum() * 0

        center_loss = self.center_loss(pred_center, target_center) * center_loss_weights
        if center_loss_weights.sum() > 0:
            center_loss = center_loss.sum() / center_loss_weights.sum()
        else:
            center_loss = center_loss.sum() * 0

        return [center_loss, offset_loss]


class OffsetLossLaplace(torch.nn.Module):
    def __init__(self):
        super(OffsetLossLaplace, self).__init__()
        self.field_names = ['offset']

    def forward(self, preds, targets):
        offset_targets, offset_weights, instance = targets

        pred_h, pred_w = preds.shape[2], preds.shape[3]
        h, w = offset_targets.shape[2], offset_targets.shape[3]

        #preds = F.interpolate(input=preds, size=(h, w), mode='bilinear',
        #                      align_corners=True)
        offset_targets = F.interpolate(input=offset_targets, size=(pred_h, pred_w),
                                       mode='bilinear', align_corners=True)
        offset_weights = F.interpolate(input=offset_weights, size=(pred_h, pred_w),
                                       mode='nearest')
        scale = (h - 1) // (pred_h - 1)

        offset_targets /= scale

        pred_y = preds[:, 0, :, :]
        pred_x = preds[:, 1, :, :]
        pred_sigma = preds[:, 2, :, :]

        #reg_masks = offset_weights.bool()
        reg_masks = torch.ones_like(offset_weights)

        loss = laplace_loss(
            torch.masked_select(pred_y, reg_masks),
            torch.masked_select(pred_x, reg_masks),
            torch.masked_select(pred_sigma, reg_masks),
            torch.masked_select(offset_targets[:, 0, :, :], reg_masks),
            torch.masked_select(offset_targets[:, 1, :, :], reg_masks),
        )

        if offset_weights.sum() > 0:
            loss = loss / reg_masks.sum() / 10
        else:
            loss = loss * 0

        return [loss]


class OffsetLoss(torch.nn.Module):
    def __init__(self, reg_loss_name='l1', hard_mining=False):
        super(OffsetLoss, self).__init__()

        self.field_names = ['offset']

        if reg_loss_name == 'smoothl1':
            self.loss = torch.nn.SmoothL1Loss(reduction='none')
        elif reg_loss_name == 'l2' or reg_loss_name == 'mse':
            self.loss = torch.nn.MSELoss(reduction='none')
        else:
            self.loss = torch.nn.L1Loss(reduction='none')

        self.hard_mining = hard_mining
        self.n_sigma = 1
        self.delta = 5

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        offset_targets, offset_weights, instance = targets
        target_offset = offset_targets[:, 0:2, ...]
        target_h, target_w = target_offset.shape[2], target_offset.shape[3]

        if isinstance(preds, list):
            loss = 0.
            for ii, pred in enumerate(preds):
                pred_h, pred_w = pred.shape[2], pred.shape[3]

                pred = F.interpolate(pred, size=(target_h, target_w),
                                     mode='bilinear', align_corners=True)
                scale = (target_h - 1) // (pred_h - 1)
                pred = pred * scale
                loss += self.loss(pred, target_offset) * offset_weights
            loss /= len(preds)
        else:
            loss = self.loss(preds, target_offset) * offset_weights

        if offset_weights.sum() > 0:
            weighted_loss = loss.sum() / offset_weights.sum() / 10
        else:
            weighted_loss = loss.sum() * 0

        return [weighted_loss]


class SemanticParsingLoss(torch.nn.Module):
    def __init__(self, ignore_index=255, only_present=True):
        super(SemanticParsingLoss, self).__init__()

        self.field_names = ['pdf']
        self.ignore_index = ignore_index
        self.only_present = only_present

        self.ce = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.bce = torch.nn.BCELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        segm_targets, edge_targets, binary_weights = targets

        h, w = segm_targets.size(1), segm_targets.size(2)
        if len(preds) <= 3:
            segm_preds = preds[0]
            edge_preds = preds[1] if len(preds) >= 2 else None

            # cross entropy loss
            segm_preds = F.interpolate(input=segm_preds, size=(h, w),
                                       mode='bilinear', align_corners=True)
            edge_preds = F.interpolate(input=edge_preds, size=(h, w),
                                       mode='bilinear', align_corners=True)

            # ============= compute semantic segmentation loss ================
            valid = binary_weights > 0
            segm_targets = segm_targets[valid, ...]
            segm_preds = segm_preds[valid, ...]

            loss_ce = self.ce(segm_preds, segm_targets.long())

            # lovasz loss
            segm_preds = F.softmax(input=segm_preds, dim=1)
            loss_lovasz = lovasz_softmax_flat(
                *flatten_probas(segm_preds, segm_targets, self.ignore_index),
                only_present=self.only_present)

            # ============= compute edge loss ================
            loss_edge = self.bce(edge_preds, edge_targets.float())

            return [loss_ce + loss_lovasz + loss_edge]
        else:
            loss = 0.
            for ii, pred in enumerate(preds):
                # cross entropy loss
                segm_preds = F.interpolate(input=pred, size=(h, w),
                                           mode='bilinear', align_corners=True)

                # ========= compute semantic segmentation loss =============
                valid = binary_weights > 0
                segm_targets_ = segm_targets[valid, ...]
                segm_preds_ = segm_preds[valid, ...]

                loss_ce = self.ce(segm_preds_, segm_targets_.long())

                if ii == 0:
                    # lovasz loss
                    segm_preds_ = F.softmax(input=segm_preds_, dim=1)
                    loss_lovasz = lovasz_softmax_flat(
                        *flatten_probas(segm_preds_, segm_targets_, self.ignore_index),
                        only_present=self.only_present)
                    loss_ce += loss_lovasz

                loss += loss_ce
            loss /= len(preds)
            return [loss]


class CascadeLoss(torch.nn.Module):
    def __init__(self, ignore_index=255, reg_loss_name='l1'):
        super(CascadeLoss, self).__init__()
        self.field_names = ['cascade']
        self.semantic_parsing_loss = SemanticParsingLoss(ignore_index=ignore_index)
        self.offset_regression_loss = OffsetLoss(reg_loss_name=reg_loss_name)

    def forward(self, preds: torch.Tensor, targets:torch.Tensor):
        return 0


class Bce(torch.nn.Module):
    def __init__(self, *, focal_gamma=0.0, detach_focal=False):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.detach_focal = detach_focal

    def forward(self, x, t):  # pylint: disable=arguments-differ
        t_zeroone = t.clone()
        t_zeroone[t_zeroone > 0.0] = 1.0
        # x = torch.clamp(x, -20.0, 20.0)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            x, t_zeroone, reduction='none')
        bce = torch.clamp(bce, 0.02, 5.0)  # 0.02 -> -3.9, 0.01 -> -4.6, 0.001 -> -7, 0.0001 -> -9

        if self.focal_gamma != 0.0:
            pt = torch.exp(-bce)
            focal = (1.0 - pt)**self.focal_gamma
            if self.detach_focal:
                focal = focal.detach()
            bce = focal * bce

        weight_mask = t_zeroone != t
        bce[weight_mask] = bce[weight_mask] * t[weight_mask]

        return bce


class ScaleLoss(torch.nn.Module):
    def __init__(self, b, *, low_clip=0.0, relative=False):
        super().__init__()
        self.b = b
        self.low_clip = low_clip
        self.relative = relative

    def forward(self, logs, t):  # pylint: disable=arguments-differ
        loss = torch.nn.functional.l1_loss(
            torch.exp(logs),
            t,
            reduction='none',
        )
        loss = torch.clamp(loss, self.low_clip, 5.0)

        loss = loss / self.b
        if self.relative:
            loss = loss / (1.0 + t)

        return loss

#def laplace_loss(x1, x2, logb, t1, t2, *, weight=None, norm_low_clip=0.0):
#    """Loss based on Laplace Distribution.
#    Loss for a single two-dimensional vector (x1, x2) with radial
#    spread b and true (t1, t2) vector.
#    """
#
#    # left derivative of sqrt at zero is not defined, so prefer torch.norm():
#    # https://github.com/pytorch/pytorch/issues/2421
#    # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
#    norm = (torch.stack((x1, x2)) - torch.stack((t1, t2))).norm(dim=0)
#    norm = torch.clamp(norm, norm_low_clip, 5.0)
#
#    # constrain range of logb
#    # low range constraint: prevent strong confidence when overfitting
#    # high range constraint: force some data dependence
#    # logb = 3.0 * torch.tanh(logb / 3.0)
#    logb = torch.clamp_min(logb, -3.0)
#
#    # ln(2) = 0.694
#    losses = logb + (norm + 0.1) * torch.exp(-logb)
#    if weight is not None:
#        losses = losses * weight
#    return losses

def laplace_loss(x1, x2, logb, t1, t2, weight=None):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """

    # left derivative of sqrt at zero is not defined, so prefer torch.norm():
    # https://github.com/pytorch/pytorch/issues/2421
    # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    norm = (torch.stack((x1, x2)) - torch.stack((t1, t2))).norm(dim=0)

    # constrain range of logb
    logb = 3.0 * torch.tanh(logb / 3.0)

    losses = 0.694 + logb + norm * torch.exp(-logb)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


def l1_loss(x1, x2, _, t1, t2, weight=None):
    """L1 loss.

    Loss for a single two-dimensional vector (x1, x2)
    true (t1, t2) vector.
    """
    losses = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


def logl1_loss(logx, t, **kwargs):
    """Swap in replacement for functional.l1_loss."""
    return torch.nn.functional.l1_loss(
        logx, torch.log(t), **kwargs)


def margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    max_r = torch.min((torch.stack(max_r1, max_r2, max_r3, max_r4)), axis=0)
    m0 = torch.isfinite(max_r)
    x = x[:, m0]
    t = t[:, m0]
    max_r = max_r[m0]

    # m1 = (x - t).norm(p=1, dim=0) > max_r
    # x = x[:, m1]
    # t = t[:, m1]
    # max_r = max_r[m1]

    norm = (x - t).norm(dim=0)
    m2 = norm > max_r

    return torch.sum(norm[m2] - max_r[m2])


def quadrant(xys):
    q = torch.zeros((xys.shape[1],), dtype=torch.long)
    q[xys[0, :] < 0.0] += 1
    q[xys[1, :] < 0.0] += 2
    return q


def quadrant_margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    diffs = x - t
    qs = quadrant(diffs)
    norms = diffs.norm(dim=0)

    m1 = norms[qs == 0] > max_r1[qs == 0]
    m2 = norms[qs == 1] > max_r2[qs == 1]
    m3 = norms[qs == 2] > max_r3[qs == 2]
    m4 = norms[qs == 3] > max_r4[qs == 3]

    return (
        torch.sum(norms[qs == 0][m1] - max_r1[qs == 0][m1]) +
        torch.sum(norms[qs == 1][m2] - max_r2[qs == 1][m2]) +
        torch.sum(norms[qs == 2][m3] - max_r3[qs == 2][m3]) +
        torch.sum(norms[qs == 3][m4] - max_r4[qs == 3][m4])
    )


class SmoothL1Loss(object):
    r_smooth = 0.0

    def __init__(self, *, scale_required=True):
        self.scale = None
        self.scale_required = scale_required

    def __call__(self, x1, x2, _, t1, t2, weight=None):
        """L1 loss.

        Loss for a single two-dimensional vector (x1, x2)
        true (t1, t2) vector.
        """
        if self.scale_required and self.scale is None:
            raise Exception
        if self.scale is None:
            self.scale = 1.0

        r = self.r_smooth * self.scale
        d = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
        smooth_regime = d < r

        smooth_loss = 0.5 / r[smooth_regime] * d[smooth_regime] ** 2
        linear_loss = d[smooth_regime == 0] - (0.5 * r[smooth_regime == 0])
        losses = torch.cat((smooth_loss, linear_loss))

        if weight is not None:
            losses = losses * weight

        self.scale = None
        return torch.sum(losses)


class MultiHeadLoss(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas):
        super(MultiHeadLoss, self).__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):  # pylint: disable=arguments-differ
        if isinstance(head_fields, dict):
            head_fields_ = []
            if 'pose' in head_fields:
                head_fields_ += head_fields['pose']
            if 'semantic' in head_fields:
                head_fields_ += [head_fields['semantic']]
            if 'offset' in head_fields:
                head_fields_ += [head_fields['offset']]
            if 'center' in head_fields:
                head_fields_ += [head_fields['center']]
            head_fields = head_fields_
        if isinstance(head_targets, dict):
            head_targets_ = []
            if 'cif' in head_targets:
                head_targets_ += [head_targets['cif']]
            if 'caf' in head_targets:
                head_targets_ += [head_targets['caf']]
            if 'caf25' in head_targets:
                head_targets_ += [head_targets['caf25']]
            if 'semantic' in head_targets:
                head_targets_ += [head_targets['semantic']]
            if 'offset' in head_targets:
                head_targets_ += [head_targets['offset']]
            if 'center' in head_targets:
                head_targets_ += [head_targets['center']]
            head_targets = head_targets_

        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        assert self.task_sparsity_weight == 0.0  # TODO implement
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, flat_head_losses)
                       if l is not None]
        total_loss = sum(loss_values) if loss_values else None

        return total_loss, flat_head_losses


class MultiHeadLossAutoTune(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None):
        """Auto-tuning multi-head less.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.sparse_task_parameters = sparse_task_parameters

        self.log_sigmas = torch.nn.Parameter(
            torch.zeros((len(lambdas),), dtype=torch.float64),
            requires_grad=True,
        )

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.log_sigmas)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.log_sigmas.exp()]}

    def forward(self, *args):
        head_fields, head_targets = args

        if isinstance(head_fields, dict):
            head_fields_ = []
            if 'pose' in head_fields:
                head_fields_ += head_fields['pose']
            if 'semantic' in head_fields:
                head_fields_ += [head_fields['semantic']]
            if 'offset' in head_fields:
                head_fields_ += [head_fields['offset']]
            if 'center' in head_fields:
                head_fields_ += [head_fields['center']]
            head_fields = head_fields_
        if isinstance(head_targets, dict):
            head_targets_ = []
            if 'cif' in head_targets:
                head_targets_ += [head_targets['cif']]
            if 'caf' in head_targets:
                head_targets_ += [head_targets['caf']]
            if 'caf25' in head_targets:
                head_targets_ += [head_targets['caf25']]
            if 'semantic' in head_targets:
                head_targets_ += [head_targets['semantic']]
            if 'offset' in head_targets:
                head_targets_ += [head_targets['offset']]
            if 'center' in head_targets:
                head_targets_ += [head_targets['center']]
            head_targets = head_targets_

        LOG.debug('losses = %d, fields = %d, targets = %d',
                  len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields), "{}, {}".format(len(self.losses), len(head_fields))
        assert len(self.losses) <= len(head_targets), "{}, {}".format(len(self.losses), len(head_targets))
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses), '{} {}'.format(len(self.lambdas), len(flat_head_losses))
        assert len(self.log_sigmas) == len(flat_head_losses)
        loss_values = [lam * l / (2.0 * (log_sigma.exp() ** 2))
                       for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                       if l is not None]
        auto_reg = [lam * log_sigma
                    for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                    if l is not None]
        total_loss = sum(loss_values) + sum(auto_reg) if loss_values else None

        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(
                # torch.norm(param, p=1)
                param.abs().max(dim=1)[0].clamp(min=1e-6).sum()
                for param in self.sparse_task_parameters
            )
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss

        return total_loss, flat_head_losses


#class CompositeLoss(torch.nn.Module):
#    background_weight = 1.0
#    focal_gamma = 1.0
#    b_scale = 1.0
#    margin = False
#
#    def __init__(self, head_net: heads.CompositeField, regression_loss):
#        super().__init__()
#        self.n_vectors = head_net.meta.n_vectors
#        self.n_scales = head_net.meta.n_scales
#
#        LOG.debug('%s: n_vectors = %d, n_scales = %d, margin = %s',
#                  head_net.meta.name, self.n_vectors, self.n_scales, self.margin)
#
#        self.confidence_loss = Bce(focal_gamma=self.focal_gamma, detach_focal=True)
#        self.regression_loss = regression_loss or laplace_loss
#        self.scale_losses = torch.nn.ModuleList([ScaleLoss(self.b_scale, low_clip=0.0)
#                                                 for _ in range(self.n_scales)])
#        self.field_names = (
#            ['{}.c'.format(head_net.meta.name)] +
#            ['{}.vec{}'.format(head_net.meta.name, i + 1) for i in range(self.n_vectors)] +
#            ['{}.scales{}'.format(head_net.meta.name, i + 1) for i in range(self.n_scales)]
#        )
#        if self.margin:
#            self.field_names += ['{}.margin{}'.format(head_net.meta.name, i + 1)
#                                 for i in range(self.n_vectors)]
#
#        self.bce_blackout = None
#        self.previous_losses = None
#
#    def _confidence_loss(self, x_confidence, target_confidence):
#        bce_masks = torch.isnan(target_confidence).bitwise_not_()
#        if not torch.any(bce_masks):
#            return None
#
#        # TODO assumes one confidence
#        x_confidence = x_confidence[:, :, 0]
#
#        batch_size = x_confidence.shape[0]
#        LOG.debug('batch size = %d', batch_size)
#
#        if self.bce_blackout:
#            x_confidence = x_confidence[:, self.bce_blackout]
#            bce_masks = bce_masks[:, self.bce_blackout]
#            target_confidence = target_confidence[:, self.bce_blackout]
#
#        LOG.debug('BCE: x = %s, target = %s, mask = %s',
#                  x_confidence.shape, target_confidence.shape, bce_masks.shape)
#        bce_target = torch.masked_select(target_confidence, bce_masks)
#        x_confidence = torch.masked_select(x_confidence, bce_masks)
#        ce_loss = self.confidence_loss(x_confidence, bce_target)
#        if self.background_weight != 1.0:
#            bce_weight = torch.ones_like(bce_target, requires_grad=False)
#            bce_weight[bce_target == 0] *= self.background_weight
#            ce_loss = ce_loss * bce_weight
#
#        ce_loss = ce_loss.sum() / (batch_size * 1000.)
#
#        return ce_loss
#
#    def _localization_loss(self, x_regs, x_logbs, target_regs):
#        batch_size = target_regs[0].shape[0]
#
#        reg_losses = []
#        for i, target_reg in enumerate(target_regs):
#            reg_masks = torch.isnan(target_reg[:, :, 0]).bitwise_not_()
#            if not torch.any(reg_masks):
#                reg_losses.append(None)
#                continue
#
#            reg_losses.append(self.regression_loss(
#                torch.masked_select(x_regs[:, :, i, 0], reg_masks),
#                torch.masked_select(x_regs[:, :, i, 1], reg_masks),
#                torch.masked_select(x_logbs[:, :, i], reg_masks),
#                torch.masked_select(target_reg[:, :, 0], reg_masks),
#                torch.masked_select(target_reg[:, :, 1], reg_masks),
#                norm_low_clip=0.0,
#            ).sum() / (batch_size * 100.0))
#
#        return reg_losses
#
#    def _scale_losses(self, x_scales, target_scales):
#        assert x_scales.shape[2] == len(target_scales)
#
#        batch_size = x_scales.shape[0]
#        return [
#            sl(
#                torch.masked_select(x_scales[:, :, i], torch.isnan(target_scale).bitwise_not_()),
#                torch.masked_select(target_scale, torch.isnan(target_scale).bitwise_not_()),
#            ).sum() / (100. * batch_size)
#            for i, (sl, target_scale) in enumerate(zip(self.scale_losses, target_scales))
#        ]
#
#    def _margin_losses(self, x_regs, target_regs, *, target_confidence):
#        if not self.margin:
#            return []
#
#        reg_masks = target_confidence > 0.5
#        if not torch.any(reg_masks):
#            return [None for _ in target_regs]
#
#        batch_size = reg_masks.shape[0]
#        margin_losses = []
#        for x_reg, target_reg in zip(x_regs, target_regs):
#            margin_losses.append(quadrant_margin_loss(
#                torch.masked_select(x_reg[:, :, 0], reg_masks),
#                torch.masked_select(x_reg[:, :, 1], reg_masks),
#                torch.masked_select(target_reg[:, :, 0], reg_masks),
#                torch.masked_select(target_reg[:, :, 1], reg_masks),
#                torch.masked_select(target_reg[:, :, 2], reg_masks),
#                torch.masked_select(target_reg[:, :, 3], reg_masks),
#                torch.masked_select(target_reg[:, :, 4], reg_masks),
#                torch.masked_select(target_reg[:, :, 5], reg_masks),
#            ) / (100.0 * batch_size))
#        return margin_losses
#
#    def forward(self, *args):
#        LOG.debug('loss for %s', self.field_names)
#
#        x, t = args
#
#        x = [xx.double() for xx in x]
#        t = [tt.double() for tt in t]
#
#        x_confidence, x_regs, x_logbs, x_scales = x
#
#        assert len(t) == 1 + self.n_vectors + self.n_scales
#        running_t = iter(t)
#        target_confidence = next(running_t)
#        target_regs = [next(running_t) for _ in range(self.n_vectors)]
#        target_scales = [next(running_t) for _ in range(self.n_scales)]
#
#        ce_loss = self._confidence_loss(x_confidence, target_confidence)
#        reg_losses = self._localization_loss(x_regs, x_logbs, target_regs)
#        scale_losses = self._scale_losses(x_scales, target_scales)
#        margin_losses = self._margin_losses(x_regs, target_regs,
#                                            target_confidence=target_confidence)
#
#        all_losses = [ce_loss] + reg_losses + scale_losses + margin_losses
#        if not all(torch.isfinite(l).item() if l is not None else True for l in all_losses):
#            raise Exception('found a loss that is not finite: {}, prev: {}'
#                            ''.format(all_losses, self.previous_losses))
#        self.previous_losses = [float(l.item()) if l is not None else None for l in all_losses]
#
#        return all_losses

class CompositeLoss(torch.nn.Module):
    background_weight = 1.0
    focal_gamma = 1.0
    margin = False

    def __init__(self, head_net: heads.CompositeField, regression_loss):
        super(CompositeLoss, self).__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d, margin = %s',
                  head_net.meta.name, self.n_vectors, self.n_scales, self.margin)

        self.regression_loss = regression_loss or laplace_loss
        self.field_names = (
            ['{}.c'.format(head_net.meta.name)] +
            ['{}.vec{}'.format(head_net.meta.name, i + 1) for i in range(self.n_vectors)] +
            ['{}.scales{}'.format(head_net.meta.name, i + 1) for i in range(self.n_scales)]
        )
        if self.margin:
            self.field_names += ['{}.margin{}'.format(head_net.meta.name, i + 1)
                                 for i in range(self.n_vectors)]

        self.bce_blackout = None

    def _confidence_loss(self, x_confidence, target_confidence):
        bce_masks = torch.isnan(target_confidence).bitwise_not_()
        if not torch.any(bce_masks):
            return None

        # TODO assumes one confidence
        x_confidence = x_confidence[:, :, 0]

        batch_size = x_confidence.shape[0]
        LOG.debug('batch size = %d', batch_size)

        if self.bce_blackout:
            x_confidence = x_confidence[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            target_confidence = target_confidence[:, self.bce_blackout]

        LOG.debug('BCE: x = %s, target = %s, mask = %s',
                  x_confidence.shape, target_confidence.shape, bce_masks.shape)
        bce_target = torch.masked_select(target_confidence, bce_masks)
        bce_weight = 1.0
        x_confidence = torch.masked_select(x_confidence, bce_masks)
        if self.background_weight != 1.0:
            bce_weight = torch.ones_like(bce_target, requires_grad=False)
            bce_weight[bce_target == 0] *= self.background_weight
        elif self.focal_gamma != 0.0:
            bce_weight = torch.empty_like(bce_target, requires_grad=False)
            bce_weight[bce_target == 1] = x_confidence[bce_target == 1]
            bce_weight[bce_target == 0] = -x_confidence[bce_target == 0]
            bce_weight = (1.0 + torch.exp(bce_weight)).pow(-self.focal_gamma)
        ce_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            x_confidence,
            bce_target,
            # weight=bce_weight,
            reduction='none',
        ) * bce_weight).sum() / (1000.0 * batch_size)

        return ce_loss

    def _localization_loss(self, x_regs, x_logbs, target_regs):
        batch_size = target_regs[0].shape[0]

        reg_losses = []
        for i, target_reg in enumerate(target_regs):
            reg_masks = torch.isnan(target_reg[:, :, 0]).bitwise_not_()
            if not torch.any(reg_masks):
                reg_losses.append(None)
                continue

            reg_losses.append(self.regression_loss(
                torch.masked_select(x_regs[:, :, i, 0], reg_masks),
                torch.masked_select(x_regs[:, :, i, 1], reg_masks),
                torch.masked_select(x_logbs[:, :, i], reg_masks),
                torch.masked_select(target_reg[:, :, 0], reg_masks),
                torch.masked_select(target_reg[:, :, 1], reg_masks),
                weight=0.1,
            ) / (100.0 * batch_size))

        return reg_losses

    @staticmethod
    def _scale_losses(x_scales, target_scales):
        assert x_scales.shape[2] == len(target_scales)

        batch_size = x_scales.shape[0]
        return [
            logl1_loss(
                torch.masked_select(x_scales[:, :, i], torch.isnan(target_scale).bitwise_not_()),
                torch.masked_select(target_scale, torch.isnan(target_scale).bitwise_not_()),
                reduction='sum',
            ) / (100.0 * batch_size)
            for i, target_scale in enumerate(target_scales)
        ]

    def _margin_losses(self, x_regs, target_regs, *, target_confidence):
        if not self.margin:
            return []

        reg_masks = target_confidence > 0.5
        if not torch.any(reg_masks):
            return [None for _ in target_regs]

        batch_size = reg_masks.shape[0]
        margin_losses = []
        for x_reg, target_reg in zip(x_regs, target_regs):
            margin_losses.append(quadrant_margin_loss(
                torch.masked_select(x_reg[:, :, 0], reg_masks),
                torch.masked_select(x_reg[:, :, 1], reg_masks),
                torch.masked_select(target_reg[:, :, 0], reg_masks),
                torch.masked_select(target_reg[:, :, 1], reg_masks),
                torch.masked_select(target_reg[:, :, 2], reg_masks),
                torch.masked_select(target_reg[:, :, 3], reg_masks),
                torch.masked_select(target_reg[:, :, 4], reg_masks),
                torch.masked_select(target_reg[:, :, 5], reg_masks),
            ) / (100.0 * batch_size))
        return margin_losses

    def forward(self, *args):
        LOG.debug('loss for %s', self.field_names)

        x, t = args

        x = [xx.double() for xx in x]
        t = [tt.double() for tt in t]

        x_confidence, x_regs, x_logbs, x_scales = x

        assert len(t) == 1 + self.n_vectors + self.n_scales
        running_t = iter(t)
        target_confidence = next(running_t)
        target_regs = [next(running_t) for _ in range(self.n_vectors)]
        target_scales = [next(running_t) for _ in range(self.n_scales)]

        ce_loss = self._confidence_loss(x_confidence, target_confidence)
        reg_losses = self._localization_loss(x_regs, x_logbs, target_regs)
        scale_losses = self._scale_losses(x_scales, target_scales)
        margin_losses = self._margin_losses(x_regs, target_regs,
                                            target_confidence=target_confidence)

        return [ce_loss] + reg_losses + scale_losses + margin_losses


def cli(parser):
    group = parser.add_argument_group('losses')
    group.add_argument('--lambdas', default=None, type=float, nargs='+',
                       help='prefactor for head losses')
    group.add_argument('--r-smooth', type=float, default=SmoothL1Loss.r_smooth,
                       help='r_{smooth} for SmoothL1 regressions')
    group.add_argument('--use-cascade', default=False)
    group.add_argument('--regression-loss', default='laplace',
                       choices=['smoothl1', 'smootherl1', 'l1', 'laplace'],
                       help='type of regression loss')
    group.add_argument('--offset-regression-loss', default='l1',
                       choices=['smoothl1', 'l1', 'mse', 'laplace'],
                       help='type of offset regression loss')
    group.add_argument('--background-weight', default=CompositeLoss.background_weight, type=float,
                       help='BCE weight where ground truth is background')
    group.add_argument('--focal-gamma', default=CompositeLoss.focal_gamma, type=float,
                       help='when > 0.0, use focal loss with the given gamma')
    group.add_argument('--margin-loss', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--offset-hard-mining', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--auto-tune-mtl', default=False, action='store_true',
                       help='use Kendall\'s prescription for adjusting the multitask weight')
    assert MultiHeadLoss.task_sparsity_weight == MultiHeadLossAutoTune.task_sparsity_weight
    group.add_argument('--task-sparsity-weight',
                       default=MultiHeadLoss.task_sparsity_weight, type=float,
                       help='[experimental]')


def configure(args):
    # apply for CompositeLoss
    CompositeLoss.background_weight = args.background_weight
    CompositeLoss.focal_gamma = args.focal_gamma
    CompositeLoss.margin = args.margin_loss

    # MultiHeadLoss
    MultiHeadLoss.task_sparsity_weight = args.task_sparsity_weight
    MultiHeadLossAutoTune.task_sparsity_weight = args.task_sparsity_weight

    # SmoothL1
    SmoothL1Loss.r_smooth = args.r_smooth


def factory_from_args(args, head_nets):
    return factory(
        head_nets,
        args.lambdas,
        reg_loss_name=args.regression_loss,
        offset_reg_loss_name=args.offset_regression_loss,
        device=args.device,
        auto_tune_mtl=args.auto_tune_mtl,
        offset_hard_mining=args.offset_hard_mining,
        use_cascade=args.use_cascade
    )


# pylint: disable=too-many-branches
def factory(head_nets, lambdas, *,
            reg_loss_name=None, device=None,
            offset_reg_loss_name=None,
            auto_tune_mtl=False,
            offset_hard_mining=False,
            use_cascade=False):
    if isinstance(head_nets[0], (list, tuple)):
        return [factory(hn, lam,
                        reg_loss_name=reg_loss_name,
                        device=device)
                for hn, lam in zip(head_nets, lambdas)]

    if reg_loss_name == 'smoothl1':
        reg_loss = SmoothL1Loss()
    elif reg_loss_name == 'l1':
        reg_loss = l1_loss
    elif reg_loss_name == 'laplace':
        reg_loss = laplace_loss
    elif reg_loss_name is None:
        reg_loss = laplace_loss
    else:
        raise Exception('unknown regression loss type {}'.format(reg_loss_name))

    #if use_cascade:
    #    cascade_loss = CascadeLoss()
    #else:
    sem_loss = SemanticParsingLoss()

    center_loss = CenterLoss()

    if 'laplace' not in offset_reg_loss_name:
        offset_loss = OffsetLoss(offset_reg_loss_name,
                                 hard_mining=offset_hard_mining)
    else:
        offset_loss = OffsetLossLaplace()

    sparse_task_parameters = None
    if MultiHeadLoss.task_sparsity_weight:
        sparse_task_parameters = []
        for head_net in head_nets:
            if getattr(head_net, 'sparse_task_parameters', None) is not None:
                sparse_task_parameters += head_net.sparse_task_parameters
            elif isinstance(head_net, heads.CompositeFieldFused):
                sparse_task_parameters.append(head_net.conv.weight)
            else:
                raise Exception('unknown l1 parameters for given head: {} ({})'
                                ''.format(head_net.meta.name, type(head_net)))

    losses = []
    for head_net in head_nets:
        if isinstance(head_net.meta, heads.ParsingMeta):
            losses.append(sem_loss)
        elif isinstance(head_net.meta, heads.OffsetMeta):
            losses.append(offset_loss)
        elif isinstance(head_net.meta, heads.CascadeMeta):
            #losses.append(sem_loss)
            losses.append(offset_loss)
        elif isinstance(head_net.meta, heads.CenterMeta):
            losses.append(center_loss)
        else:
            losses.append(CompositeLoss(head_net, reg_loss))

    if auto_tune_mtl:
        loss = MultiHeadLossAutoTune(losses, lambdas,
                                     sparse_task_parameters=sparse_task_parameters)
    else:
        loss = MultiHeadLoss(losses, lambdas)

    if device is not None:
        loss = loss.to(device)

    return loss

