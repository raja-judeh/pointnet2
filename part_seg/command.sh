#!/bin/sh

#eval `rmpm_do env --env-format embed_sh software.common.tensorflow==1.13.1`
source /volume/USERSTORE/jude_ra/master_thesis/pointnet2/virtual_envs/tf_GPU_py3.4/bin/activate
python train_multipoints.py --model pointnet2_multipoints_sn_lfp --num_point_pairs 100 --batch_size 12 --max_epoch 200 --log_dir /volume/USERSTORE/jude_ra/master_thesis/pointnet2/phase5/log_folders/2nd_phase/log_trial
