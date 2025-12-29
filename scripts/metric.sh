#!/bin/bash
#SBATCH --account=ruqiz
#SBATCH --job-name=e3_1_5b
#SBATCH --partition=ai       # or the queue name you used before
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # adjust for your job
#SBATCH --cpus-per-task=14
#SBATCH --mem=40G
#SBATCH --time=01:00:00
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

python3 -m e3.main_metric