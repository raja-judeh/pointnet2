 #!/bin/sh

source /volume/USERSTORE/jude_ra/master_thesis/pointnet2/virtual_envs/tf_GPU_py3.4/bin/activate
python evaluate.py --model pointnet2_part_seg_msg --model_path /volume/USERSTORE/jude_ra/master_thesis/pointnet2/log_folders/the_log/log_msg/model.ckpt --batch_size 16 --log_dir log_folders/log_evaluation/log_msg

 
