#!/bin/bash
#SBATCH -J eval_mul_indom
#SBATCH -p debug_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -o /data/leecg1219/MAE_Recon/eval_mulsen_indomain.log
#SBATCH -e /data/leecg1219/MAE_Recon/eval_mulsen_indomain.err

eval "$(/data/leecg1219/miniconda3/bin/conda shell.bash hook)"
conda activate splatposeplus
cd /data/leecg1219/MAE_Recon

python eval.py \
  --ckpt /data/leecg1219/MAE_Recon/output_mulsen_v2_50ep/mae_final.pt \
  --normalizer /data/leecg1219/MAE_Recon/output_mulsen_v2_50ep/normalizer.npz \
  --data_root /data/leecg1219/MulSen_3DGS_COMBINED \
  --classes capsule button_cell cotton cube flat_pad light nut piggy plastic_cylinder screen screw solar_panel spring_pad toothbrush zipper \
  --anomaly_types burrs_recon stains_recon missing_recon \
  --n_iter 10

mv /data/leecg1219/MAE_Recon/output_mulsen_v2_50ep/eval_results/eval_raw_orig.npz /data/leecg1219/MAE_Recon/output_mulsen_v2_50ep/eval_results/eval_raw_mulsen_indomain.npz
mv /data/leecg1219/MAE_Recon/output_mulsen_v2_50ep/eval_results/eval_summary_orig.json /data/leecg1219/MAE_Recon/output_mulsen_v2_50ep/eval_results/eval_summary_mulsen_indomain.json
