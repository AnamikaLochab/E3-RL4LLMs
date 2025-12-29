#!/bin/bash
#SBATCH --account=ruqiz
#SBATCH --job-name=run_ood_gen
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --time=12:00:00
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
# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_a14/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_a14_id_s32_1.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32
# echo "Done at $(date)"
# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_a15/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_a15_id_s32_1.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32


TEST_DATADIR=./dataset/test.json
MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_a16/global_step_480/actor/huggingface
OUTPUT_FILE=./results/dGRPO_1_5b_1.0_a16_id_s32_1.json

python3 -m e3.main_infer \
    --model=$MODELDIR \
    --input_file=$TEST_DATADIR \
    --output_file=$OUTPUT_FILE \
    --tensor_parallel_size=1 \
    --gpu_memory_utilization=0.95 \
    --temperature=1 \
    --max_tokens=6144 \
    --n_samples=32
# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_a2/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_a2_id_s32_1.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32


# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_a1/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_a1_id_s32_1.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32



# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_r_norm_dyn_clamp_1.5/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_r_norm_dyn_clamp_1.5_id_s32_1.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32

# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_r_norm_dyn_clamp_2.0/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_r_norm_dyn_clamp_2.0_id_s32_1.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32


# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_r_norm_dyn_clamp_2.5/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_r_norm_dyn_clamp_2.5_id_s32_1.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32
# TEST_DATADIR=./dataset/all.jsonl
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_r_norm_dyn_clamp_1.5/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_r_norm_dyn_clamp_1.5_ood_s32_1.json


# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32

# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_grpo/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_grpo_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32

# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_grpo_2/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_grpo_2_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32

# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_grpo_3/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_grpo_3_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32


# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_e3/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_e3_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32

# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_e3_2/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_e3_2_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32

# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_e3_3/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_e3_3_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32


# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.00003/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dgrpo_1_5b_0.00003_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32


# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.00001_3/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_dGRPO_0.00001_3_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32

# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.00005/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dgrpo_1_5b_0.00005_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32


# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.5_seqq_norm/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dgrpo_0.5_seqq_norm_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32


# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_seqq_norm/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dgrpo_1.0_seqq_norm_id_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32

