#!/bin/bash
#SBATCH --account=ruqiz
#SBATCH --job-name=e3_1_5b
#SBATCH --partition=ai       # or the queue name you used before
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # adjust for your job
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
module purge
module load gcc/11.4.1

ENV=/scratch/gautschi/alochab/conda_envs/e3
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

echo "host=$(hostname)"
echo "CVD=${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi -L || true

python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("is_available:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name0:", torch.cuda.get_device_name(0))
print("CVD env:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
echo "Launching training at $(date)"

python3 -m e3.main_eval \
  --result_file ./results/dGRPO_1_5b_1.0_a14_id_s32_1.json  \
  --output_file ./metrics/dGRPO_1_5b_1.0_a14_id_s32_1_metrics.json \
  --model_name ../checkpoint/e3_1_5b_dGRPO_1.0_a14/global_step_480/actor/huggingface

python3 -m e3.main_eval \
  --result_file ./results/dGRPO_1_5b_1.0_a15_id_s32_1.json  \
  --output_file ./metrics/dGRPO_1_5b_1.0_a15_id_s32_1_metrics.json \
  --model_name ../checkpoint/e3_1_5b_dGRPO_1.0_a15/global_step_480/actor/huggingface

python3 -m e3.main_eval \
  --result_file ./results/dGRPO_1_5b_1.0_a16_id_s32_1.json  \
  --output_file ./metrics/dGRPO_1_5b_1.0_a16_id_s32_1_metrics.json \
  --model_name ../checkpoint/e3_1_5b_dGRPO_1.0_a16/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dGRPO_1_5b_1.0_a8_id_s32_1.json  \
#   --output_file ./metrics/dGRPO_1_5b_1.0_a8_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_1.0_a8/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dGRPO_1_5b_1.0_r_norm_dyn_clamp_1.5_id_s32_1.json \
#   --output_file ./metrics/dGRPO_1_5b_1.0_r_norm_dyn_clamp_1.5_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_1.0_r_norm_dyn_clamp_1.5/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dGRPO_1_5b_1.0_r_norm_dyn_clamp_2.0_id_s32_1.json \
#   --output_file ./metrics/dGRPO_1_5b_1.0_r_norm_dyn_clamp_2.0_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_1.0_r_norm_dyn_clamp_2.0/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dGRPO_1_5b_1.0_r_norm_dyn_clamp_2.5_id_s32_1.json \
#   --output_file ./metrics/dGRPO_1_5b_1.0_r_norm_dyn_clamp_2.5_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_1.0_r_norm_dyn_clamp_2.5/global_step_480/actor/huggingface

# # python3 -m e3.main_eval \
# #   --result_file ./results/dGRPO_1_5b_1.0_r_norm_ood_s32_1.json  \
# #   --output_file ./metrics/dGRPO_1_5b_1.0_r_norm_ood_s32_1_metrics.json \
# #   --model_name ../checkpoint/e3_1_5b_dGRPO_1.0_r_norm/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dGRPO_1_5b_0.5_r_norm_id_s32_1.json  \
#   --output_file ./metrics/dGRPO_1_5b_0.5_r_norm_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.5_r_norm/global_step_480/actor/huggingface


# python3 -m e3.main_eval \
#   --result_file ./results/dGRPO_1_5b_0.5_r_norm_ood_s32_1.json  \
#   --output_file ./metrics/dGRPO_1_5b_0.5_r_norm_ood_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.5_r_norm/global_step_480/actor/huggingface


# python3 -m e3.main_eval \
#   --result_file ./results/dgrpo_1_5b_0.00001_2_id_s32.json \
#   --output_file ./metrics/dgrpo_1_5b_0.00001_2_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.00001_2/global_step_480/actor/huggingface
# python3 -m e3.main_eval \
#   --result_file ./results/dgrpo_1_5b_0.00005_ood_s32.json \
#   --output_file ./metrics/dGRPO_1_5b_0.00005_ood_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.00005/global_step_480/actor/huggingface


# python3 -m e3.main_eval \
#   --result_file ./results/e3_1_5b_e3_id_s32.json \
#   --output_file ./metrics/e3_1_5b_e3_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_e3/global_step_480/actor/huggingface


# python3 -m e3.main_eval \
#   --result_file ./results/e3_1_5b_e3_2_id_s32.json \
#   --output_file ./metrics/e3_1_5b_e3_2_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_e3_2/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/e3_1_5b_e3_3_id_s32.json \
#   --output_file ./metrics/e3_1_5b_e3_3_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_e3_3/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dgrpo_1_5b_0.00003_id_s32.json \
#   --output_file ./metrics/dGRPO_1_5b_0.00003_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.00003/global_step_480/actor/huggingface


# python3 -m e3.main_eval \
#   --result_file ./results/e3_1_5b_dGRPO_0.00001_3_id_s32.json \
#   --output_file ./metrics/dGRPO_1_5b_0.00001_3_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.00001_3/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/e3_1_5b_dGRPO_0.00001_3_ood_s32.json \
#   --output_file ./metrics/dGRPO_1_5b_0.00001_3_ood_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.00001_3/global_step_480/actor/huggingface


# python3 -m e3.main_eval \
#   --result_file ./results/dgrpo_1_5b_0.00005_id_s32.json \
#   --output_file ./metrics/dGRPO_1_5b_0.00005_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.00005/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dgrpo_0.5_seqq_norm_id_s32.json \
#   --output_file ./metrics/dGRPO_0.5_seqq_norm_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.5_seqq_norm/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dgrpo_1.0_seqq_norm_id_s32.json \
#   --output_file ./metrics/dGRPO_1.0_seqq_norm_id_s32_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_1.0_seqq_norm/global_step_480/actor/huggingface
# python3 -m e3.main_eval \
#   --result_file ./results/dgrpo_1_5b_0.00003_s16_1.json \
#   --output_file ./metrics/dgrpo_1_5b_0.00003_s16_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.00003/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dgrpo_1_5b_0.00005_2sub_half_s16_1.json \
#   --output_file ./metrics/dgrpo_1_5b_0.00005_2sub_half_s16_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.00005_2sub_half/global_step_480/actor/huggingface

# # python3 -m e3.main_eval \
# #   --result_file ./results/dgrpo_1_5b_0.00008_s16_1.json \
# #   --output_file ./metrics/dgrpo_1_5b_0.00008_s16_1_metrics.json \
# #   --model_name ../checkpoint/e3_1_5b_dGRPO_0.00008/global_step_480/actor/huggingface

# # python3 -m e3.main_eval \
# #   --result_file ./results/dgrpo_1_5b_0.0005_s16_1.json \
# #   --output_file ./metrics/dgrpo_1_5b_0.0005_s16_1_metrics.json \
# #   --model_name ../checkpoint/e3_1_5b_dGRPO_0.0005/global_step_480/actor/huggingface

# # python3 -m e3.main_eval \
# #   --result_file ./results/tw_covvar_grpo_1_5b_0.0005_s16_1.json \
# #   --output_file ./metrics/tw_covvar_grpo_1_5b_0.0005_s16_1_metrics.json \
# #   --model_name ../checkpoint/e3_1_5b_tw_covvar_0.0005_2/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/dgrpo_1_5b_2.0_seqq_norm_s16_1.json \
#   --output_file ./metrics/dgrpo_1_5b_2.0_seqq_norm_s16_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_2.0_seqq_norm/global_step_480/actor/huggingface

# python3 -m e3.main_eval \
#   --result_file ./results/e3_1_5b_grpo_id_s16.json \
#   --output_file ./metrics//e3_1_5b_grpo_id_s16_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_grpo/global_step_480/actor/huggingface


# python3 -m e3.main_eval \
#   --result_file ./results/e3_1_5b_grpo_3_s16_1.json \
#   --output_file ./metrics/e3_1_5b_grpo_3_s16_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_grpo_3/global_step_480/actor/huggingface
#   --result_file ./results/dgrpo_1_5b_0.5_seqq_norm_s16_1.json \
#   --output_file ./metrics/dgrpo_1_5b_0.5_seqq_norm_s16_1_metrics.json \
#   --model_name ../checkpoint/e3_1_5b_dGRPO_0.5_seqq_norm/global_step_480/actor/huggingface