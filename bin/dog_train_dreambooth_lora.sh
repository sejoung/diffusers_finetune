#!/usr/bin/bash

# Using RTX 3090 or 4000 series which doesn't support faster communication speedups. Ensuring P2P and IB communications are disabled
# https://github.com/hiyouga/LLaMA-Factory/issues/2359
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="../data/dreambooth/dog"
export OUTPUT_DIR="../models/dreambooth-lora/dog"
export LOGGING_DIR="../logs/dreambooth-lora/dog"

accelerate launch ../src/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=20 \
  --seed=42 \
  --enable_xformers_memory_efficient_attention \
  --logging_dir=${LOGGING_DIR}
