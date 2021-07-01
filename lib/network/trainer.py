"""Train a neural net."""

import copy
import hashlib
import logging
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from ..metric import fast_hist, per_class_iu

LOG = logging.getLogger(__name__)


try:
    autocast = torch.cuda.amp.autocast
    import torch.cuda.amp.GradScaler as GradScaler
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def get_lip_palette():
    palette = [0, 0, 0,
               128, 0, 0,
               255, 0, 0,
               0, 85, 0,
               170, 0, 51,
               255, 85, 0,
               0, 0, 85,
               0, 119, 221,
               85, 85, 0,
               0, 85, 85,
               85, 51, 0,
               52, 86, 128,
               0, 128, 0,
               0, 0, 255,
               51, 170, 221,
               0, 255, 255,
               85, 255, 170,
               170, 255, 85,
               255, 255, 0,
               255, 170, 0]
    return palette


palette = get_lip_palette()


def adjust_learning_rate(lr, epochs, optimizer, epoch, i_iter, iters_per_epoch, method='poly'):
    if method == 'poly':
        current_step = epoch * iters_per_epoch + i_iter
        max_step = epochs * iters_per_epoch
        lr = lr * ((1 - current_step / max_step) ** 0.9)
    else:
        lr = lr
    optimizer.param_groups[0]['lr'] = lr
    return lr


class Trainer(object):
    def __init__(self, model, loss, optimizer, out, *,
                 lr_scheduler=None,
                 log_interval=10,
                 device=None,
                 fix_batch_norm=False,
                 stride_apply=1,
                 ema_decay=None,
                 train_profile=None,
                 num_class=None,
                 model_meta_data=None,
                 mixed_precision=False,
                 clip_grad_norm=0.0,
                 val_interval=1):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.out = out
        self.lr_scheduler = lr_scheduler

        self.log_interval = log_interval
        self.device = device
        self.fix_batch_norm = fix_batch_norm
        self.stride_apply = stride_apply

        self.ema_decay = ema_decay
        self.ema = None
        self.ema_restore_params = None

        self.clip_grad_norm = clip_grad_norm
        self.n_clipped_grad = 0
        self.max_norm = 0.0

        self.model_meta_data = model_meta_data

        self.mixed_precision = mixed_precision

        self.num_class = num_class
        self.val_hists = np.zeros((num_class, num_class)) if self.num_class is not None else None

        self.val_interval = val_interval

        if train_profile:
            # monkey patch to profile self.train_batch()
            self.trace_counter = 0
            self.train_batch_without_profile = self.train_batch

            def train_batch_with_profile(*args, **kwargs):
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    result = self.train_batch_without_profile(*args, **kwargs)
                print(prof.key_averages())
                self.trace_counter += 1
                tracefilename = train_profile.replace(
                    '.json', '.{}.json'.format(self.trace_counter))
                LOG.info('writing trace file %s', tracefilename)
                prof.export_chrome_trace(tracefilename)
                return result

            self.train_batch = train_batch_with_profile

        LOG.info({
           'type': 'config',
           'field_names': self.loss.field_names,
        })

    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step_ema(self):
        if self.ema is None:
            return

        for p, ema_p in zip(self.model.parameters(), self.ema):
            ema_p.mul_(1.0 - self.ema_decay).add_(p.data, alpha=self.ema_decay)

    def apply_ema(self):
        if self.ema is None:
            return

        LOG.info('applying ema')
        self.ema_restore_params = copy.deepcopy(
            [p.data for p in self.model.parameters()])
        for p, ema_p in zip(self.model.parameters(), self.ema):
            p.data.copy_(ema_p)

    def ema_restore(self):
        if self.ema_restore_params is None:
            return

        LOG.info('restoring params from before ema')
        for p, ema_p in zip(self.model.parameters(), self.ema_restore_params):
            p.data.copy_(ema_p)
        self.ema_restore_params = None

    def loop(self, train_scenes, val_scenes, epochs, start_epoch=0):
        if self.lr_scheduler is not None:
           with warnings.catch_warnings():
               warnings.simplefilter('ignore')
               for _ in range(start_epoch * len(train_scenes)):
                   self.lr_scheduler.step()

        self.epochs = epochs
        for epoch in range(start_epoch, epochs):
            self.train(train_scenes, epoch)

            if (epoch + 1) % self.val_interval == 0 or epoch + 1 == epochs:
                self.write_model(epoch + 1, epoch + 1 == epochs)
                self.val(val_scenes, epoch + 1)

    def train_batch(self, data, targets, apply_gradients=True):  # pylint: disable=method-hidden
        if self.device:
            data = data.to(self.device, non_blocking=True)

            if isinstance(targets, list):
                targets = [[t.to(self.device, non_blocking=True) for t in head] for head in targets]
            elif isinstance(targets, dict):
                for k, v in targets.items():
                    tt = {k: [t.to(self.device, non_blocking=True) for t in v]}
                    targets.update(tt)
            else:
                targets = targets.to(self.device, non_blocking=True)

        # train encoder
        outputs = self.model(data)
        loss, head_losses = self.loss(outputs, targets)

        if loss is not None:
            loss.backward()
        if self.clip_grad_norm:
            max_norm = self.clip_grad_norm / self.lr()
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm, norm_type=float('inf'))
            self.max_norm = max(float(total_norm), self.max_norm)
            if total_norm > max_norm:
                self.n_clipped_grad += 1
                print(
                    'CLIPPED GRAD NORM: total norm before clip: {}, max norm: {}'
                    ''.format(total_norm, max_norm))
        if apply_gradients:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_ema()

        return (
            float(loss.item()) if loss is not None else None,
            [float(l.item()) if l is not None else None
             for l in head_losses],
        )

    def val_batch(self, data, targets):
        if self.device:
            data = data.to(self.device, non_blocking=True)
            if isinstance(targets, list):
                targets = [[t.to(self.device, non_blocking=True) for t in head] for head in targets]
            elif isinstance(targets, dict):
                for k, v in targets.items():
                    tt = {k: [t.to(self.device, non_blocking=True) for t in v]}
                    targets.update(tt)
            else:
                targets = targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(data)
            loss, head_losses = self.loss(outputs, targets)

        if 'semantic' in outputs:
            targets = targets['semantic'][0]
            h, w = targets.shape[1], targets.shape[2]
            pred = F.interpolate(outputs['semantic'][0], size=(h, w),
                                 mode='bilinear', align_corners=True)
            hist = fast_hist(pred, targets, self.num_class)
        else:
            hist = None

        return (
            float(loss.item()) if loss is not None else None,
            [float(l.item()) if l is not None else None
             for l in head_losses],
            hist
        )

    def train(self, scenes, epoch):
        start_time = time.time()
        self.model.train()
        if self.fix_batch_norm:
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.eval()

        self.ema_restore()
        self.ema = None

        epoch_loss = 0.0
        head_epoch_losses = None
        head_epoch_counts = None
        last_batch_end = time.time()
        self.optimizer.zero_grad()
        for batch_idx, (data, target, _) in enumerate(scenes):
            preprocess_time = time.time() - last_batch_end

            #adjust_learning_rate(optimizer=self.optimizer,
            #                     epoch=epoch, epochs=self.epochs,
            #                     lr=self.lr(), i_iter=batch_idx,
            #                     iters_per_epoch=len(scenes))

            batch_start = time.time()
            apply_gradients = batch_idx % self.stride_apply == 0
            loss, head_losses = self.train_batch(data, target, apply_gradients)

            # update epoch accumulates
            if loss is not None:
                epoch_loss += loss
            if head_epoch_losses is None:
                head_epoch_losses = [0.0 for _ in head_losses]
                head_epoch_counts = [0 for _ in head_losses]
            for i, head_loss in enumerate(head_losses):
                if head_loss is None:
                    continue
                head_epoch_losses[i] += head_loss
                head_epoch_counts[i] += 1

            batch_time = time.time() - batch_start

            # write training loss
            if batch_idx % self.log_interval == 0:
                batch_info = {
                    'type': 'train',
                    'epoch': epoch, 'batch': batch_idx, 'n_batches': len(scenes),
                    'time': round(batch_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': round(self.lr(), 8),
                    'loss': round(loss, 3) if loss is not None else None,
                    'head_losses': [round(l, 3) if l is not None else None
                                    for l in head_losses],
                }
                if hasattr(self.loss, 'batch_meta'):
                    batch_info.update(self.loss.batch_meta())
                LOG.info(batch_info)

            # initialize ema
            if self.ema is None and self.ema_decay:
                self.ema = copy.deepcopy([p.data for p in self.model.parameters()])

            # update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            last_batch_end = time.time()

        self.apply_ema()
        LOG.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / len(scenes), 5),
            'head_losses': [round(l / max(1, c), 5)
                            for l, c in zip(head_epoch_losses, head_epoch_counts)],
            'time': round(time.time() - start_time, 1),
            'n_clipped_grad': self.n_clipped_grad,
            'max_norm': self.max_norm,
        })
        self.n_clipped_grad = 0
        self.max_norm = 0.0

    def val(self, scenes, epoch):
        start_time = time.time()

        # Train mode implies outputs are for losses, so have to use it here.
        self.model.train()
        if self.fix_batch_norm:
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.eval()

        epoch_loss = 0.0
        head_epoch_losses = None
        head_epoch_counts = None
        for ii, (data, target, _) in enumerate(scenes):
            loss, head_losses, hist = self.val_batch(data, target)

            if self.val_hists is not None and hist is not None:
                self.val_hists += hist

            # update epoch accumulates
            if loss is not None:
                epoch_loss += loss
            if head_epoch_losses is None:
                head_epoch_losses = [0.0 for _ in head_losses]
                head_epoch_counts = [0 for _ in head_losses]
            for i, head_loss in enumerate(head_losses):
                if head_loss is None:
                    continue
                head_epoch_losses[i] += head_loss
                head_epoch_counts[i] += 1

        eval_time = time.time() - start_time

        if self.val_hists is not None:
            mIoU = round(np.nanmean(per_class_iu(self.val_hists)) * 100, 2)
        else:
            mIoU = None

        LOG.info({
            'type': 'val-epoch',
            'epoch': epoch,
            'loss': round(epoch_loss / len(scenes), 5),
            'head_losses': [round(l / max(1, c), 5)
                            for l, c in zip(head_epoch_losses, head_epoch_counts)],
            'time': round(eval_time, 1),
            'mIoU': mIoU,
        })

    def write_model(self, epoch, final=True):
        self.model.cpu()

        if isinstance(self.model, torch.nn.DataParallel):
            LOG.debug('Writing a dataparallel model.')
            model = self.model.module
        else:
            LOG.debug('Writing a single-thread model.')
            model = self.model

        filename = '{}.epoch{:03d}'.format(self.out, epoch)
        LOG.debug('about to write model')
        torch.save({
            'model': model,
            'epoch': epoch,
            'meta': self.model_meta_data,
        }, filename)
        LOG.debug('model written')

        if final:
            sha256_hash = hashlib.sha256()
            with open(filename, 'rb') as f:
                for byte_block in iter(lambda: f.read(8192), b''):
                    sha256_hash.update(byte_block)
            file_hash = sha256_hash.hexdigest()
            outname, _, outext = self.out.rpartition('.')
            final_filename = '{}-{}.{}'.format(outname, file_hash[:8], outext)
            shutil.copyfile(filename, final_filename)

        self.model.to(self.device)
