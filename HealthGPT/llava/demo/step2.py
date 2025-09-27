import os, sys
sys.path.append("/root/autodl-tmp/HealthGPT")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import tokenizers
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
from PIL import Image
import pickle
import argparse
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
from utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square
from tqdm import tqdm
import pandas as pd
import time
import psutil
import torch.cuda
import csv
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_multimodal_model(args):
    # 修复：如果使用FlashAttention，强制使用fp16或bf16
    if args.attn_implementation == 'flash_attention_2' and args.dtype == 'FP32':
        print("Warning: FlashAttention只支持fp16和bf16，自动将dtype从FP32改为FP16")
        args.dtype = 'FP16'
    
    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)
    print(f"使用的数据类型: {model_dtype}")
    print(f"使用的注意力实现: {args.attn_implementation}")

    model = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=model_dtype  # 修复：使用计算出的model_dtype而不是硬编码的torch.float16
    ).to(device)

    from llava.peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.hlora_r,
        lora_alpha=args.hlora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=args.hlora_dropout,
        bias='none',
        task_type="CAUSAL_LM",
        lora_nums=args.hlora_nums,
    )
    model = get_peft_model(model, lora_config).to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model, args.vq_idx_nums)
    print(f"Number of new tokens added for unified task: {num_new_tokens}")

    from utils import com_vision_args
    com_vision_args.model_name_or_path = args.model_name_or_path
    com_vision_args.vision_tower = args.vit_path
    com_vision_args.version = args.instruct_template

    model.get_model().initialize_vision_modules(model_args=com_vision_args)
    model.get_vision_tower().to(dtype=model_dtype)

    model = load_weights(model, args.hlora_path)
    model.eval()
    model.to(model_dtype).cuda()
    return model, tokenizer

def generate_description_with_healthgpt(model, tokenizer, args, question, img_path):
    if img_path:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        qs = question
    conv = conversation_lib.conv_templates[args.instruct_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)
    
    if img_path:
        image = Image.open(img_path).convert('RGB')
        image = expand2square(image, tuple(int(x*255) for x in model.get_vision_tower().image_processor.image_mean))
        image_tensor = model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
    
    # 修复：使用与模型一致的数据类型
    if args.attn_implementation == 'flash_attention_2' and args.dtype == 'FP32':
        args.dtype = 'FP16'  # 确保一致性
    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)
    
    with torch.inference_mode():
        output_ids = model.base_model.model.generate(
        input_ids,
        images=image_tensor.to(dtype=model_dtype, device='cuda', non_blocking=True) if img_path else None,
        image_sizes=image.size if img_path else None,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True)
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)[:-8]
    return output_text

def concat_picture_future(question, key_features):
    user_prompt_step2 = (
        "The following clinical question was raised regarding the image:\n"
        f"Question: {question}\n\n"
        "To address this question, the following key visual indicators have been identified as important:\n"
        f"{key_features}\n\n"
        "Based on the image and the visual indicators above, provide a precise and objective description of the relevant image features.\n"
        "Describe findings such as anatomical locations, sizes, shapes, intensities, textures, or boundaries of any visible structures or abnormalities.\n"
        "Do **NOT** attempt to answer the clinical question or make diagnostic judgments—your role is strictly to describe what is seen in the image.\n"
        "Do **NOT** include any information beyond the image-based findings."
    )
    return user_prompt_step2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/root/autodl-tmp/HealthGPT/models/Phi-3-mini-4k-instruct/models--microsoft--Phi-3-mini-4k-instruct/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85")
    # 修复：默认使用FP16以避免FlashAttention兼容性问题
    parser.add_argument('--dtype', type=str, default='FP16', choices=['FP32', 'FP16', 'BF16'])
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2')
    parser.add_argument('--hlora_r', type=int, default=64)
    parser.add_argument('--hlora_alpha', type=int, default=128)
    parser.add_argument('--hlora_dropout', type=float, default=0.0)
    parser.add_argument('--hlora_nums', type=int, default=4)
    parser.add_argument('--vq_idx_nums', type=int, default=8192)
    parser.add_argument('--instruct_template', type=str, default='phi3_instruct')
    parser.add_argument('--vit_path', type=str, default="/root/autodl-tmp/HealthGPT/models/clip-vit-large-patch14-336/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
    parser.add_argument('--hlora_path', type=str, default="/root/autodl-tmp/HealthGPT/models/HealthGPT-M3/models--lintw--HealthGPT-M3/snapshots/a1184cc40b3341509e00549a2b19d2c1c8ff955c/com_hlora_weights.bin")
    parser.add_argument('--fusion_layer_path', type=str, default="/root/autodl-tmp/HealthGPT/models/HealthGPT-M3/models--lintw--HealthGPT-M3/snapshots/a1184cc40b3341509e00549a2b19d2c1c8ff955c/fusion_layer_weights.bin")
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument("--step1-file", type=str, default="/root/autodl-tmp/HealthGPT/ZDH/train_0217/step1_results.json")
    parser.add_argument("--output-file", type=str, default="/root/autodl-tmp/HealthGPT/ZDH/train_0217/step2_results.json")
    
    args = parser.parse_args()
    
    # 提前检查兼容性
    if args.attn_implementation == 'flash_attention_2' and args.dtype == 'FP32':
        print("警告: FlashAttention只支持fp16和bf16数据类型")
        print("自动将dtype从FP32修改为FP16以保证兼容性")
        args.dtype = 'FP16'
    
    # 加载模型
    print("正在加载HealthGPT模型...")
    model, tokenizer = load_multimodal_model(args)
    print("模型加载完成")
    
    # 加载第一步的结果
    print(f"正在加载第一步结果: {args.step1_file}")
    with open(args.step1_file, 'r', encoding='utf-8') as f:
        all_step1_data = json.load(f)
    
    print(f"加载了 {len(all_step1_data)} 条第一步的结果")
    
    # 筛选出step1_success为True的数据
    step1_success_data = [item for item in all_step1_data if item.get("step1_success", False) == True]
    step1_failed_data = [item for item in all_step1_data if item.get("step1_success", False) != True]
    
    print(f"第一步成功的数据: {len(step1_success_data)} 条")
    print(f"第一步失败的数据: {len(step1_failed_data)} 条")
    
    # 处理第二步 - 只处理第一步成功的数据
    print("开始处理第二步...")
    start_time = time.time()
    
    processed_results = []
    
    # 先处理第一步成功的数据
    for idx, item in enumerate(tqdm(step1_success_data, desc="处理第二步(成功项)")):
        try:
            # 生成第二步的提示
            step2_prompt = concat_picture_future(item["question"], item["key_features"])
            
            # 使用HealthGPT生成图像描述
            step2_start = time.time()
            
            if os.path.exists(item["img_path"]):
                response_step2 = generate_description_with_healthgpt(
                    model, tokenizer, args, step2_prompt, item["img_path"]
                )
                step2_success = True
                step2_error = ""
            else:
                response_step2 = ""
                step2_success = False
                step2_error = f"Image file not found: {item['img_path']}"
            
            step2_end = time.time()
            
            # 更新数据
            item.update({
                "image_description": response_step2,
                "step2_success": step2_success,
                "step2_error": step2_error,
                "step2_duration": step2_end - step2_start,
                "step2_prompt": step2_prompt
            })
            
        except Exception as e:
            print(f"处理第 {idx} 项时出错: {str(e)}")
            item.update({
                "image_description": "",
                "step2_success": False,
                "step2_error": str(e),
                "step2_duration": 0,
                "step2_prompt": ""
            })
        
        processed_results.append(item)
    
    # 将第一步失败的数据也加入结果，并标记第二步为跳过
    for item in step1_failed_data:
        item.update({
            "image_description": "",
            "step2_success": False,
            "step2_error": "Step1 failed - skipped step2",
            "step2_duration": 0,
            "step2_prompt": ""
        })
        processed_results.append(item)
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n第二步处理完成，总耗时: {total_duration:.2f}秒")
    
    # 统计结果
    step2_success_count = sum(1 for item in processed_results if item.get("step2_success", False) == True)
    step2_failed_count = len(processed_results) - step2_success_count
    step2_processed_count = len(step1_success_data)  # 实际处理的数量
    
    # 保存结果
    print("正在保存结果...")
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"结果已保存到: {args.output_file}")
    print(f"总数据量: {len(processed_results)}")
    print(f"第一步成功(需要处理第二步): {len(step1_success_data)}")
    print(f"第一步失败(跳过第二步): {len(step1_failed_data)}")
    print(f"第二步成功: {step2_success_count}")
    print(f"第二步失败: {step2_failed_count}")
    if step2_processed_count > 0:
        print(f"第二步成功率: {step2_success_count/step2_processed_count*100:.1f}% (基于实际处理的数据)")
        print(f"平均处理时间: {total_duration/step2_processed_count:.2f}秒/条 (仅计算实际处理的数据)")
    else:
        print("没有需要处理第二步的数据")

if __name__ == "__main__":
    main()