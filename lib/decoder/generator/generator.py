from abc import abstractmethod
import logging
import multiprocessing
import sys
import time

import torch

from ... import visualizer

LOG = logging.getLogger(__name__)


class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]


class Generator:
    def __init__(self, worker_pool=None):
        if worker_pool is None or worker_pool == 0:
            worker_pool = DummyPool()
        if isinstance(worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', worker_pool)
            assert not sys.platform.startswith('win'), (
                'not supported, use --decoder-workers=0 '
                'on windows'
            )
            worker_pool = multiprocessing.Pool(worker_pool)

        self.worker_pool = worker_pool

        self.last_decoder_time = 0.0
        self.last_nn_time = 0.0

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ('worker_pool',)
        }

    @staticmethod
    def fields_batch(model, image_batch, *, device=None):
        """From image batch to field batch."""
        start = time.time()

        def apply(f, items):
            """Apply f in a nested fashion to all items that are not list or tuple."""
            if items is None:
                return None
            if isinstance(items, (list, tuple)):
                return [apply(f, i) for i in items]
            return f(items)

        with torch.no_grad():
            if device is not None:
                image_batch = image_batch.to(device, non_blocking=True)

            with torch.autograd.profiler.record_function('model'):
                heads = model(image_batch)

            # to numpy
            with torch.autograd.profiler.record_function('tonumpy'):
                if isinstance(heads, list):
                    heads = apply(lambda x: x.cpu().numpy(), heads)
                elif isinstance(heads, dict):
                    if 'pose' in heads:
                        heads.update({'pose': apply(lambda x: x.cpu().numpy(), heads['pose'])})
                    if 'semantic' in heads:
                        if isinstance(heads['semantic'], list):
                            if len(heads['semantic']) == 2:
                                heads['edge'] = heads['semantic'][1].cpu().numpy()
                            heads.update({'semantic': heads['semantic'][0].cpu().numpy()})
                        else:
                            heads.update({'semantic': heads['semantic'].cpu().numpy()})
                    if 'offset' in heads:
                        if isinstance(heads['offset'], list):
                            heads.update({'offset': heads['offset'][0].cpu().numpy()})
                        else:
                            heads.update({'offset': heads['offset'].cpu().numpy()})
                    if 'vote' in heads:
                        heads.update({'vote': heads['vote'].cpu().numpy()})
                    if 'center' in heads:
                        heads.update({'center_offset': heads['center']['center_offset'].cpu().numpy()})
                        heads.update({'center': heads['center']['center'].cpu().numpy()})

        LOG.debug('nn processing time: %.3fs', time.time() - start)
        return heads

    @abstractmethod
    def __call__(self, fields, *, initial_annotations=None):
        """For single image, from fields to annotations."""
        raise NotImplementedError()

    def batch(self, model, image_batch, *, device=None):
        def apply(f, items):
            """Apply f in a nested fashion to all items that are not list or tuple."""
            if items is None:
                return None
            if isinstance(items, (list, tuple)):
                return [apply(f, i) for i in items]
            return f(items)
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        fields_batchs = self.fields_batch(model, image_batch, device=device)
        self.last_nn_time = time.perf_counter() - start_nn

        if isinstance(fields_batchs, dict):
            fields_batch = fields_batchs['pose']
        elif isinstance(fields_batchs, list):
            fields_batch = fields_batchs

        # index by frame (item in batch)
        head_iter = apply(iter, fields_batch)
        fields_batch = []
        while True:
            try:
                fields_batch.append(apply(next, head_iter))
            except StopIteration:
                break

        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
            image_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        start_decoder = time.perf_counter()
        result = self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, image_batch))
        self.last_decoder_time = time.perf_counter() - start_decoder

        LOG.debug('time: nn = %.3fs, dec = %.3fs', self.last_nn_time, self.last_decoder_time)

        all_results = result
        if isinstance(fields_batchs, dict):
            all_results = {'pose': result}
            for key, value in fields_batchs.items():
                if key != 'pose':
                    all_results[key] = value

        return all_results

    def _mappable_annotations(self, fields, debug_image):
        if debug_image is not None:
            visualizer.BaseVisualizer.processed_image(debug_image)

        return self(fields)
