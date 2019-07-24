#!/bin/sh

source /net/rmc-lx0185/home_local/virtual_envs/tf_GPU_py3.4/bin/activate
python evaluate.py --model pointnet2_part_seg_relcoor --batch_size 16 --model_path log_folders/log_relcoor/model.ckpt --log_dir log_folders/log_eval_relcoor_rotated

 
