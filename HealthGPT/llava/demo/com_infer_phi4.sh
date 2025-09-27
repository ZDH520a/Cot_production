#!/bin/bash

MODEL_NAME_OR_PATH="/root/autodl-tmp/HealthGPT/models/Phi-4"
VIT_PATH="/root/autodl-tmp/HealthGPT/models/clip-vit-large-patch14-336/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
HLORA_PATH="/root/autodl-tmp/HealthGPT/models/com_hlora_weights_phi4.bin"

python3 com_infer_phi4.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP16" \
    --hlora_r "32" \
    --hlora_alpha "64" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi4_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --question "You are a specialized pathologist, please describe what you see microscopically on the neuroblastoma pathology image." \
    --img_path "/root/autodl-tmp/HealthGPT/neuroblastoma/gallery1500/all/D_13_147.jpg"
