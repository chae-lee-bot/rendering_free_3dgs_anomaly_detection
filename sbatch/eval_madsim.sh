#!/bin/bash
#SBATCH -J eval_madsim
#SBATCH -p debug_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -o /data/leecg1219/MAE_Recon/eval_madsim_v2_50ep.log
#SBATCH -e /data/leecg1219/MAE_Recon/eval_madsim_v2_50ep.err

eval "$(/data/leecg1219/miniconda3/bin/conda shell.bash hook)"
conda activate splatposeplus
cd /data/leecg1219/MAE_Recon

python eval.py \
  --ckpt /data/leecg1219/MAE_Recon/output_madsim_v2_15d_200ep/mae_final.pt \
  --normalizer /data/leecg1219/MAE_Recon/output_madsim_v2_15d_200ep/normalizer.npz \
  --data_root /data/leecg1219/MAD_Sim_v2_COMBINED \
  --classes 01Gorilla 02Unicorn 03Mallard 04Turtle 05Whale 06Bird 07Owl 08Sabertooth 09Swan 10Sheep 11Pig 12Zalika 13Pheonix 14Elephant 15Parrot 16Cat 17Scorpion 18Obesobeso 19Bear 20Puppy \
  --anomaly_types burrs_recon stains_recon missing_recon \
  --subsample --n_iter 10

mv /data/leecg1219/MAE_Recon/output_madsim_v2_15d_200ep/eval_results /data/leecg1219/MAE_Recon/output_madsim_v2_15d_200ep/eval_results_madsim
