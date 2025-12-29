#!/bin/bash
#SBATCH --account=ruqiz
#SBATCH --job-name=run_ood_gen
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=400G
#SBATCH --time=20:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# --- 1. Modules (same as interactive) ---
module purge
module load gcc/11.4.1

# --- 2. CUDA (matches your flash-attn / torch build) ---
export CUDA_HOME=/apps/gautschi/cuda/12.6
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- 3. Use the EXISTING e3 env directly (NO conda) ---
export ENV=/scratch/gautschi/alochab/conda_envs/e3
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1

export TMPDIR=/scratch/gautschi/alochab/pip_tmp
export PIP_CACHE_DIR=/scratch/gautschi/alochab/.pip/cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# Tokens / HF cache
export HF_TOKEN=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
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/scratch/gautschi/alochab/huggingface_cache
mkdir -p "$HF_HOME"

echo "=== RUNTIME ENV ==="
echo "which python: $(which python)"
python -V || true
echo "which accelerate: $(which accelerate || echo 'none')"
echo "CUDA_HOME      : $CUDA_HOME"
echo "ENV            : $ENV"
echo "==================="

# quick sanity check: torch + HF imports from e3
python - <<'PY'
import sys, torch
print("PY:", sys.executable)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
PY
echo "Launching training at $(date)"
bash ./scripts/train_1_5b.sh
# echo "Launching eval at $(date)"
# bash ./scripts/infer.sh
echo "Done at $(date)"

# #!/bin/bash
# #SBATCH --account=ruqiz
# #SBATCH --job-name=run_ood_gen
# #SBATCH --partition=ai       # or the queue name you used before
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:2                 # adjust for your job
# #SBATCH --cpus-per-task=28
# #SBATCH --mem=100G
# #SBATCH --time=08:00:00
# #SBATCH -o logs/%x-%j.out
# #SBATCH -e logs/%x-%j.err

# # --- 1. Load modules exactly as in interactive run ---
# module load gcc/11.4.1

# # --- 2. CUDA setup (matches flash-attn & torch build) ---
# export CUDA_HOME=/apps/gautschi/cuda/12.6
# export CUDA_PATH=$CUDA_HOME
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# export TMPDIR=/scratch/gautschi/alochab/pip_tmp
# export PIP_CACHE_DIR=/scratch/gautschi/alochab/.pip/cache
# mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
# export HF_TOKEN=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
# export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
# export HF_HOME=/scratch/gautschi/alochab/huggingface_cache
# # --- 3. Activate your env ---
# source /scratch/gautschi/alochab/miniconda_clean/etc/profile.d/conda.sh
# conda activate /scratch/gautschi/alochab/conda_envs/e3

# # echo "Launching training at $(date)"
# # bash ./scripts/train_1_5b.sh
# echo "Launching eval at $(date)"
# bash ./scripts/infer_ood.sh
# echo "Done at $(date)"