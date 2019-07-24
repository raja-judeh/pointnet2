#!/bin/sh

source /net/rmc-lx0185/home_local/virtual_envs/tf_GPU_py3.4/bin/activate
python train_rotated.py --model pointnet2_part_seg --batch_size 16 --log_dir log_folders/log_rotated --num_batches_test 200

 
