#!/bin/sh

#eval `rmpm_do env --env-format embed_sh software.common.tensorflow==1.13.1`
source /volume/USERSTORE/jude_ra/master_thesis/pointnet2/virtual_envs/tf_GPU_py3.4/bin/activate
python evaluate_occlusion.py --model pointnet2_part_seg_msg_one_hot --model_path /volume/USERSTORE/jude_ra/master_thesis/pointnet2/log_folders/log_msg_onehot/model.ckpt --recentering True --log_dir /volume/USERSTORE/jude_ra/master_thesis/pointnet2/log_folders/log_evaluation/log_occ_trial
