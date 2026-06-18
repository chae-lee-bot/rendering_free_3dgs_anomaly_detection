#!/bin/bash
#SBATCH -J train_madsim_15d_200ep
#SBATCH -p debug_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -o /data/leecg1219/MAE_Recon/train_madsim_v2_15d_200ep.log
#SBATCH -e /data/leecg1219/MAE_Recon/train_madsim_v2_15d_200ep.err

eval "$(/data/leecg1219/miniconda3/bin/conda shell.bash hook)"
conda activate splatposeplus
cd /data/leecg1219/MAE_Recon/

python train.py
