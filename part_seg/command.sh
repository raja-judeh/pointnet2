#!/bin/sh

source /volume/USERSTORE/jude_ra/master_thesis/pointnet2/virtual_envs/tf_GPU_py3.4/bin/activate
python train_rotated.py --model pointnet2_part_seg_msg_transform --batch_size 12 --max_epoch 200 --log_dir /volume/USERSTORE/jude_ra/master_thesis/pointnet2/log_folders/log_msg_transform_rotated

