#!/usr/bin/bash

# Using RTX 3090 or 4000 series which doesn't support faster communication speedups. Ensuring P2P and IB communications are disabled
# https://github.com/hiyouga/LLaMA-Factory/issues/2359
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export LOGGING_DIR="logs"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="../data/dreambooth/dog"
export OUTPUT_DIR="../models/dreambooth/dog"

accelerate launch ../src/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --enable_xformers_memory_efficient_attention \
  --logging_dir=$LOGGING_DIR
