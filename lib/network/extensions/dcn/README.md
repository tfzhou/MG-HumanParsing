# Deformable-ConvNets-V2 in PyTorch

This repo is an implementation of [Deformable Convolution V2](https://arxiv.org/abs/1811.11168).
Ported from the original [MXNet implementation](https://github.com/msracver/Deformable-ConvNets/tree/master/DCNv2_op).

Refer to [mmdetection branch](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/mmdetection) in this repo for a complete framework. Results of DCNv2 based on mmdetection code base can be found at [model zoo](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/MODEL_ZOO.md#deformable-conv-v2). Many thanks to [mmdetection](https://github.com/open-mmlab/mmdetection) for their strong and clean framework.

Operators in master branch are compatible with pytorch_v0.4.1. For operators on pytorch v1.0.0 (implemented by [Jiarui Xu](https://github.com/xvjiarui)), please refer to [pytorch_1.0.0 branch](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0).

Thanks to [Kai Chen](https://github.com/hellock) and other contributors from mmlab, DCNv2 is now included in the official mmdetection repo based on the master branch of this one. It is now written with the new cpp extension apis and it supports both PyTorch 0.4.1 and 1.0, with some minor speed and memory optimization. Results and models can be found at https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md#deformable-convolution-v2.

## Build

```
sh make.sh
```

See `test.py` and `test_modulated.py` for example usage.

## Notice

This repo provides the deformable conv layer which can reproduce the results in the Deformable ConvNets v2 paper. The major changes are as follows:

* To better handle occasions where sampling locations are outside of the image boundary.

    In the previous operator, if the sampling location is outside of the feature map boundary, its sampled value would be zero. Thus, the gradient with respect to learnable offset would be zero. We found such a scheme may deteriate the performance in ImageNet classification (perhaps because the feature maps are of low resolution). For object detection on COCO, both the previous and the updated operators deliver the same results.

    In the new operator, if the sampling location is within one pixel outside of the feature map boundary, bilinear sampling would also be applied. And gradient with respect to learnable offset can be non zero for such locations. This is implemented by padding zeros (by one row/column) outside of the boundaries of feature maps, and performing bilinear sampling on the padded feature maps.


* The efficiency of processing multiple images in a mini-batch is considerably improved.

    Both the previous and the updated operators follow the following computation pipeline (illustrated by a 3x3 deformable convolution with input data of NxCxHxW and output data of NxC'xHxW):

      for i in range(N/S):
          step 1 (slicing): slicing the input data at the batch dimension from i*S to (i+1)*S, input (NxCxHxW) -> sliced input (SxCxHxW)
          step 2 (deformable im2col): sliced input (SxCxHxW)+sliced offset (Sx18xHxW) -> column (Cx9xSxHxW)
          step 3 (MatMul&reshape): weight matrix (C'x 9C) * column (9CxSHW) -> temp sliced output (C'xSxHxW) -> sliced output (SxC'xHxW)
          step 4 (Merge): merge sliced output to form the whole output data (NxC'xHxW) 
      end

    In the previous operator, S is fixed as 1. In the updated operator, S can be set by the *im2col_step* parameter, whose default value is min(N, 64). The updated operator is significantly faster than the existing one when the image batch size is large.
