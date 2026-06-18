#!/bin/bash
#SBATCH -J train_shapenet_50ep
#SBATCH -p debug_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -o /data/leecg1219/MAE_Recon/train_shapenet_v2_50ep.log
#SBATCH -e /data/leecg1219/MAE_Recon/train_shapenet_v2_50ep.err

eval "$(/data/leecg1219/miniconda3/bin/conda shell.bash hook)"
conda activate splatposeplus
cd /data/leecg1219/MAE_Recon

python train.py \
  --data_root /data/leecg1219/Anomaly_ShapeNet_COMBINED \
  --output_dir /data/leecg1219/MAE_Recon/output_shapenet_v2_50ep \
  --classes ashtray0 bag0 bottle0 bottle3 bowl0 bowl3 bucket0 cap0 cap3 cup0 eraser0 headset0 helmet0 helmet2 jar0 microphone0 shelf0 tap0 vase0 vase5 \
  --epochs 50
