#!/bin/bash
#SBATCH -J train_mulsen_50ep
#SBATCH -p debug_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -o /data/leecg1219/MAE_Recon/train_mulsen_v2_50ep.log
#SBATCH -e /data/leecg1219/MAE_Recon/train_mulsen_v2_50ep.err

eval "$(/data/leecg1219/miniconda3/bin/conda shell.bash hook)"
conda activate splatposeplus
cd /data/leecg1219/MAE_Recon

python train.py \
  --data_root /data/leecg1219/MulSen_3DGS_COMBINED \
  --output_dir /data/leecg1219/MAE_Recon/output_mulsen_v2_50ep \
  --classes capsule button_cell cotton cube flat_pad light nut piggy plastic_cylinder screen screw solar_panel spring_pad toothbrush zipper \
  --epochs 50
