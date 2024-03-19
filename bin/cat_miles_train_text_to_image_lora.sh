#!/usr/bin/bash

# Using RTX 3090 or 4000 series which doesn't support faster communication speedups. Ensuring P2P and IB communications are disabled
# https://github.com/hiyouga/LLaMA-Factory/issues/2359
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="../data/full-finetune/cat"
export OUTPUT_DIR="../models/lora/miles"

accelerate launch --mixed_precision="fp16"  ../src/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="A photo of a cat in a bucket" \
  --validation_epochs=10 \
  --seed=42
