#!/bin/sh

source /net/rmc-lx0185/home_local/virtual_envs/tf_GPU_py3.4/bin/activate
python train.py --model pointnet2_part_seg_transform --batch_size 16 --log_dir log_folders/log_transform

 
