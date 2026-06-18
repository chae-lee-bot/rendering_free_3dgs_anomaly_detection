#!/bin/bash
#SBATCH -J eval_shp_indom
#SBATCH -p debug_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -o /data/leecg1219/MAE_Recon/eval_shapenet_indomain.log
#SBATCH -e /data/leecg1219/MAE_Recon/eval_shapenet_indomain.err

eval "$(/data/leecg1219/miniconda3/bin/conda shell.bash hook)"
conda activate splatposeplus
cd /data/leecg1219/MAE_Recon

python eval.py \
  --ckpt /data/leecg1219/MAE_Recon/output_shapenet_v2_50ep/mae_final.pt \
  --normalizer /data/leecg1219/MAE_Recon/output_shapenet_v2_50ep/normalizer.npz \
  --data_root /data/leecg1219/Anomaly_ShapeNet_COMBINED \
  --classes ashtray0 bag0 bottle0 bottle3 bowl0 bowl3 bucket0 cap0 cap3 cup0 eraser0 headset0 helmet0 helmet2 jar0 microphone0 shelf0 tap0 vase0 vase5 \
  --anomaly_types burrs_recon stains_recon missing_recon \
  --n_iter 10

mv /data/leecg1219/MAE_Recon/output_shapenet_v2_50ep/eval_results/eval_raw_orig.npz /data/leecg1219/MAE_Recon/output_shapenet_v2_50ep/eval_results/eval_raw_shapenet_indomain.npz
mv /data/leecg1219/MAE_Recon/output_shapenet_v2_50ep/eval_results/eval_summary_orig.json /data/leecg1219/MAE_Recon/output_shapenet_v2_50ep/eval_results/eval_summary_shapenet_indomain.json
