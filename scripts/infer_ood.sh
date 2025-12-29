# TEST_DATADIR=./dataset/test.json
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_r_norm/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_r_norm_id_s32_1.json

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
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_r_norm/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_1.0_r_norm_ood_s32_1.json

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
MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.5_r_norm/global_step_480/actor/huggingface
OUTPUT_FILE=./results/dGRPO_1_5b_0.5_r_norm_id_s32_1.json

python3 -m e3.main_infer \
    --model=$MODELDIR \
    --input_file=$TEST_DATADIR \
    --output_file=$OUTPUT_FILE \
    --tensor_parallel_size=1 \
    --gpu_memory_utilization=0.95 \
    --temperature=1 \
    --max_tokens=6144 \
    --n_samples=32

# TEST_DATADIR=./dataset/all.jsonl
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.5_r_norm/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dGRPO_1_5b_0.5_r_norm_ood_s32_1.json

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
# MODELDIR=../checkpoint/e3_1_5b_grpo_2/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_grpo_2_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_grpo_3/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_grpo_3_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_e3/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_e3_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_e3_2/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_e3_2_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_e3_3/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_e3_3_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.00003/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dgrpo_1_5b_0.00003_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.00001/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dgrpo_1_5b_0.00001_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.00005/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dgrpo_1_5b_0.00005_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_0.5_seqq_norm/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dgrpo_0.5_seqq_norm_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_dGRPO_1.0_seqq_norm/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/dgrpo_1.0_seqq_norm_ood_s32.json

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
# MODELDIR=../checkpoint/e3_1_5b_grpo/global_step_480/actor/huggingface
# OUTPUT_FILE=./results/e3_1_5b_grpo_ood_s32.json

# python3 -m e3.main_infer \
#     --model=$MODELDIR \
#     --input_file=$TEST_DATADIR \
#     --output_file=$OUTPUT_FILE \
#     --tensor_parallel_size=1 \
#     --gpu_memory_utilization=0.95 \
#     --temperature=1 \
#     --max_tokens=6144 \
#     --n_samples=32