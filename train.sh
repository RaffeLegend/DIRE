#!/bin/bash

#SBATCH -p gpulowmed -N 1 -n 16
#SBATCH -J Dire
#SBATCH -o log.out
#SBATCH -e error.err
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00

# define the path of the output images
# OUTPUT="output_images"

# make sure the folder of output exist
# mkdir -p $OUTPUT_DIR

# execute shell
EXP_NAME="lsun_adm"
DATASETS="lsun_adm"
DATASETS_TEST="lsun_adm"

python train.py  --exp_name dire --datasets_path /mnt/data2/users/hilight/yiwei/dataset/DomainSet --ckpt_dir ./dire_finetune
