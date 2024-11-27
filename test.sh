#!/bin/bash

#SBATCH -p gpulowmed -N 1 -n 16
#SBATCH -J val
#SBATCH -o val.out
#SBATCH -e val.err
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00

# define the path of the output images
# OUTPUT="output_images"

# make sure the folder of output exist
# mkdir -p $OUTPUT_DIR

# execute shell

EXP_NAME="dire_domain"
# CKPT="./checkpoint/lsun_adm.pth"
CKPT="./dire_finetune/model_epoch_best.pth"
DATASETS_TEST="/mnt/data2/users/hilight/yiwei/dataset/DomainSet/"
python test.py --gpus 0 --ckpt $CKPT --exp_name $EXP_NAME --datasets_path $DATASETS_TEST
