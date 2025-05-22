#!/usr/bin/env bash

#================================================#
#  Non-Streaming ASR/Speech Translation Script   #
#  Replace placeholders with your paths          #
#================================================#

# Project root directory
ROOT="{your_project_root}"

# Hugging Face cache directory
HF_CACHE_DIR="{your_hf_cache_dir}"

# Experiments output directory
EXP_DIR="{your_experiment_dir}"

# LibriTTS dataset directory
LIBRITTS_DIR="{your_libritts_dir}"

# Pretrained ASR/LM checkpoint path
BEST_LM_CKPT="{your_best_asr_lm_checkpoint}"

# Source global settings (optional)
source "$ROOT/scripts/global.sh"

# Add project root to PYTHONPATH
export PYTHONPATH="$ROOT"

# Number of GPUs to use (passed as first argument)
num_gpus=${1:-1}
echo "Using $num_gpus GPUs"

# Model and training hyperparameters
striding_factor=320
lr=1e-4
batch_size=32
grad_accum_every=1
max_seconds=10
max_steps=500000

dataset_name="libritts"
use_nuggets=True
nugget_compress_rate=18
nuggets_pretrain_steps=6000
window_size=256

# Construct run name and output directory
suffix="lr_${lr}_bz_${batch_size}_grad_acc_${grad_accum_every}_max_len_${max_seconds}s_nug_pretrain_${nuggets_pretrain_steps}"
run_name="${dataset_name}_w2v_lm_nuggets_${nugget_compress_rate}_${suffix}"
output_dir="$EXP_DIR/ckpt/soundstream/$run_name"
mkdir -p "$output_dir"

# Launch ASR/LM training with nugget module
poetry run torchrun \
  --nproc_per_node="$num_gpus" \
  --rdzv-endpoint "0.0.0.0:25001" \
  "$ROOT/src/train_w2v.py" \
  --model_path "$BEST_LM_CKPT" \
  --run_name "$run_name" \
  --cache_dir "$HF_CACHE_DIR" \
  --tokenizer_vocab_path "$EXP_DIR/$dataset_name/tokenizer/vocab.json" \
  --use_ctc False \
  --use_char_ctc False \
  --use_lm True \
  --disable_nuggets False \
  --learning_rate "$lr" \
  --use_nuggets "$use_nuggets" \
  --nuggets_pretrain_steps "$nuggets_pretrain_steps" \
  --nugget_compress_rate "$nugget_compress_rate" \
  --nugget_window_size "$window_size" \
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