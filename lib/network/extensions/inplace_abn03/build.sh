#!/bin/bash

# Configuration
CUDA_GENCODE="\
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_70,code=compute_70
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_50,code=sm_50"


cd src
nvcc -I/usr/local/cuda-8.0/include --expt-extended-lambda -O3 -c -o bn.o bn.cu -x cu -Xcompiler -fPIC -std=c++11 ${CUDA_GENCODE}
cd ..
