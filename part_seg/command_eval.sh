#!/bin/sh

#eval `rmpm_do env --env-format embed_sh software.common.tensorflow==1.13.1`
source /volume/USERSTORE/jude_ra/master_thesis/pointnet2/virtual_envs/tf_GPU_py3.4/bin/activate

#python evaluate_one_hot.py --model pointnet2_part_seg_msg_one_hot --model_path /volume/USERSTORE/jude_ra/master_thesis/pointnet2/phase1/log_folders/log_msg_onehot/model.ckpt --log_dir /volume/USERSTORE/jude_ra/master_thesis/pointnet2/phase5/log_folders/2nd_phase/log_evaluation/log_msg

python evaluate_multipoints.py --model pointnet2_multipoints_sn_cp --model_path /volume/USERSTORE/jude_ra/master_thesis/pointnet2/phase5/log_folders/2nd_phase/log_pploss_100_sn_cp_sum/model.ckpt --log_dir /volume/USERSTORE/jude_ra/master_thesis/pointnet2/phase5/log_folders/2nd_phase/log_evaluation/log_pploss_100_sn_cp_sum
