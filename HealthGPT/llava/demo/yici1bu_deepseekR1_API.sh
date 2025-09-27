#!/bin/bash

MODEL_NAME_OR_PATH="/root/autodl-tmp/HealthGPT/models/Phi-3-mini-4k-instruct/models--microsoft--Phi-3-mini-4k-instruct/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85"
VIT_PATH="/root/autodl-tmp/HealthGPT/models/clip-vit-large-patch14-336/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
HLORA_PATH="/root/autodl-tmp/HealthGPT/models/HealthGPT-M3/models--lintw--HealthGPT-M3/snapshots/a1184cc40b3341509e00549a2b19d2c1c8ff955c/com_hlora_weights.bin"
FUSION_LAYER_PATH="/root/autodl-tmp/HealthGPT/models/HealthGPT-M3/models--lintw--HealthGPT-M3/snapshots/a1184cc40b3341509e00549a2b19d2c1c8ff955c/fusion_layer_weights.bin"

python3 yici1bu_deepseekR1_API.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP16" \
    --hlora_r "64" \
    --hlora_alpha "128" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --fusion_layer_path "$FUSION_LAYER_PATH" \
