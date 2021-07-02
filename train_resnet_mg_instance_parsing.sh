#!/usr/bin/env sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m lib.train_v2 \
  --lr=0.05 \
  --epochs=100 \
  --lr-decay 80 \
  --lr-decay-epochs=20 \
  --batch-size=64 \
  --loader-workers=8 \
  --square-edge=385 \
  --weight-decay=1e-5 \
  --update-batchnorm-runningstatistics \
  --dataset densepose \
  --lr-warm-up-epochs 1 \
  --basenet=resnet50deeplab \
  --num-classes 15 \
  --lambdas 1 1 0.2   1 1 1 0.2 0.2    1 1 1 0.2 0.2   1  1\
  --headnets cif caf caf25 pdf offset \
  --auto-tune-mtl \
  --with-edge \
  --offset-hard-mining \
