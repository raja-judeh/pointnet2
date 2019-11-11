#!/bin/sh

#eval `rmpm_do env --env-format embed_sh software.common.tensorflow==1.13.1`
source /volume/USERSTORE/jude_ra/master_thesis/pointnet2/virtual_envs/tf_GPU_py3.4/bin/activate
python evaluate_one_hot.py --model pointnet2_random_triples --model_path /volume/USERSTORE/jude_ra/master_thesis/pointnet2/phase3/log_folders/log_rand_tree_triples/model.ckpt --log_dir /volume/USERSTORE/jude_ra/master_thesis/pointnet2/phase3/log_folders/log_evaluation/log_rand_tree_triples
