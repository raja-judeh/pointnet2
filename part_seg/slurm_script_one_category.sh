#!/bin/sh

source /net/rmc-lx0185/home_local/virtual_envs/tf_GPU_py3.4/bin/activate
python train_one_category.py --model pointnet2_part_seg_relcoor --batch_size 16 --input_cat Airplane --log_dir log_folders/log_airplane_relcoor --max_epoch 400

