#!/usr/bin/bash

# Using RTX 3090 or 4000 series which doesn't support faster communication speedups. Ensuring P2P and IB communications are disabled
# https://github.com/hiyouga/LLaMA-Factory/issues/2359
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export LOGGING_DIR="logs"
xport MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="../data/dreambooth/cat"
export CLASS_DIR="../data/dreambooth/cat-class"
export OUTPUT_DIR="../models/dreambooth-lora/miles"

accelerate launch ../src/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks cat" \
  --class_prompt="a photo of a cat" \
  --with_prior_preservation --prior_loss_weight=0.5 \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --validation_prompt="A photo of sks cat in a bucket" \
  --num_class_images=200 \
  --validation_epochs=10 \
  --seed=42 \
  --enable_xformers_memory_efficient_attention \
  --logging_dir=$LOGGING_DIR
