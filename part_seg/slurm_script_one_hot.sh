#!/bin/sh

source /net/rmc-lx0185/home_local/virtual_envs/tf_GPU_py3.4/bin/activate
python train_one_hot.py --model pointnet2_part_seg_msg_one_hot_relcoor --log_dir log_folders/log_msg_one_hot_relcoor --batch_size 12
