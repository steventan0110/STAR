#!/usr/bin/env bash

#========================================#
#  Simultaneous Speech-to-Text Script    #
#  Replace placeholders with your paths  #
#========================================#

# Project root directory
ROOT="{your_project_root}"

# Hugging Face cache directory
HF_CACHE_DIR="{your_hf_cache_dir}"

# Experiments output directory
EXP_DIR="{your_experiment_dir}"

# LibriTTS dataset directory
LIBRITTS_DIR="{your_libritts_dir}"

# Pretrained nugget checkpoint path
BEST_NUGGET_CKPT="{your_best_nugget_checkpoint}"

# Source global settings (optional)
source "$ROOT/scripts/global.sh"

# Add project root to PYTHONPATH
export PYTHONPATH="$ROOT"

# Number of GPUs to use (passed as first argument)
num_gpus=${1:-1}
echo "Using $num_gpus GPUs"

# Training hyperparameters
lr=1e-4
batch_size=32
grad_accum_every=1
max_seconds=10
max_steps=200000
dataset_name="libritts"
use_nuggets=True
nuggets_pretrain_steps=200000

# Construct run name and output directory
suffix="lr_${lr}_bz_${batch_size}_grad_acc_${grad_accum_every}_max_len_${max_seconds}s"
run_name="${dataset_name}_w2v_simul_nugget_block_attn_${suffix}"
output_dir="$EXP_DIR/ckpt/soundstream/$run_name"

# Create output directory
mkdir -p "$output_dir"

# Launch training
poetry run torchrun \
  --nproc_per_node="$num_gpus" \
  --rdzv-endpoint 0.0.0.0:25001 \
  "$ROOT/src/train_simul.py" \
  --model_path "$BEST_NUGGET_CKPT" \
  --use_nuggets "$use_nuggets" \
  --use_cif False \
  --tokenizer_vocab_path "$EXP_DIR/$dataset_name/tokenizer/vocab.json" \
  --run_name "$run_name" \
  --cache_dir "$HF_CACHE_DIR" \
  --learning_rate "$lr" \
  --nuggets_pretrain_steps "$nuggets_pretrain_steps" \
  --max_steps "$max_steps" \
  --log_freq 50 \
  --eval_freq 600 \
  --save_freq 2000 \
  --train_file "$LIBRITTS_DIR/train.txt" \
  --valid_file "$LIBRITTS_DIR/valid.txt" \
  --test_file "$LIBRITTS_DIR/test.txt" \
  --batch_size "$batch_size" \
  --grad_accum_every "$grad_accum_every" \
  --max_seconds "$max_seconds" \
  --target_sample_hz 16000 \
  --output_dir "$output_dir" \
  --num_gpus "$num_gpus"