#!/bin/sh

#eval `rmpm_do env --env-format embed_sh software.common.tensorflow==1.13.1`
source /volume/USERSTORE/jude_ra/master_thesis/pointnet2/virtual_envs/tf_GPU_py3.4/bin/activate
python train_one_hot.py --model pointnet2_sp --batch_size 12 --max_epoch 200 --log_dir /volume/USERSTORE/jude_ra/master_thesis/pointnet2/phase3/log_folders/log_trial

