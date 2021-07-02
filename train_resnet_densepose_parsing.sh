#!/usr/bin/env sh

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m lib.train_v2 \
  --lr=7e-3 \
  --epochs=150 \
  --lr-decay 120 \
  --lr-decay-epochs=20 \
  --batch-size=64 \
  --loader-workers=8 \
  --square-edge=473 \
  --lambdas 1 \
  --weight-decay=1e-5 \
  --update-batchnorm-runningstatistics \
  --dataset densepose \
  --lr-warm-up-epochs 0 \
  --basenet=resnet50deeplab \
  --headnets pdf \
  --num-classes 15 \
  --extended-scale \


