#!/bin/sh

source /net/rmc-lx0185/home_local/virtual_envs/tf_GPU_py3.4/bin/activate
python train.py --model pointnet2_sem_seg --batch_size 16
