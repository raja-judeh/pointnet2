#!/bin/sh

#eval `rmpm_do env --env-format embed_sh software.common.tensorflow==1.13.1`
source /volume/USERSTORE/jude_ra/master_thesis/pointnet2/virtual_envs/tf_GPU_py3.4/bin/activate
python train_multi_gpu.py --model pointnet2_cls_triplenet --batch_size 16 --num_gpus 2 --max_epoch 300 --log_dir /volume/USERSTORE/jude_ra/master_thesis/pointnet2/cls/log_folders/log_cls_triplenet_allPolygons4 --normal
