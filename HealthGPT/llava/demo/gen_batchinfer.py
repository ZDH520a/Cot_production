import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append("/root/autodl-tmp/HealthGPT")
import copy, re
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
import pandas as pd
import time
import psutil
import torch.cuda
from tqdm import tqdm
import csv


def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--dtype', type=str, default='FP32')
    parser.add_argument('--attn_implementation', type=str, default=None)
    parser.add_argument('--hlora_r', type=int, default=16)
    parser.add_argument('--hlora_alpha', type=int, default=32)
    parser.add_argument('--hlora_dropout', type=float, default=0.0)
    parser.add_argument('--hlora_nums', type=int, default=4)
    parser.add_argument('--vq_idx_nums', type=int, default=1024)
    parser.add_argument('--instruct_template', type=str, default='phi3_instruct')
    parser.add_argument('--vit_path', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--hlora_path', type=str, default=None)
    parser.add_argument('--fusion_layer_path', type=str, default=None)
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1100)
    parser.add_argument('--save_path', type=str, default='./example.jpg')
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/HealthGPT/question_gen.jsonl")
    parser.add_argument("--occupancy_output_file", type=str, default="/root/autodl-tmp/HealthGPT/occupancy_gen_batchinfer.csv")
    

    args = parser.parse_args()

    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)

    model = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        #attn_implementation=args.attn_implementation,
        attn_implementation="flash_attention_2",  # 显式启用 flash-attention
        torch_dtype=model_dtype
    ) #模型初始化后立即将其移动到 GPU 上，否则会warning：flash-attention 被正确启用，但模型初始化时没有在 GPU 上。虽然这不会阻止 flash-attention 的运行，但可能会导致性能下降。

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
    model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model, args.vq_idx_nums)
    print(f"Number of new tokens added for unified task: {num_new_tokens}")

    from utils import gen_vision_args
    gen_vision_args.model_name_or_path = args.model_name_or_path
    gen_vision_args.vision_tower = args.vit_path
    gen_vision_args.version = args.instruct_template

    model.get_model().initialize_vision_modules(model_args=gen_vision_args)
    model.get_vision_tower().to(dtype=model_dtype)

    model = load_weights(model, args.hlora_path, args.fusion_layer_path)
    model.eval()
    
    # 确保模型在 GPU 上
    model = model.to(model_dtype).cuda()  # 确保模型在 GPU 上
    #model.to(model_dtype).cuda()

    from taming_transformers.idx2img import idx2img
    
    ##################################################################################
    #从这里开始要循环遍历实现批量推理
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    inference_records = []
    
    for line in tqdm(questions):
        #question = args.question
        #img_path = args.img_path
        
         # 在每次推理前记录显存占用、CPU和内存使用情况
        start_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # 显存使用量（MB）
        start_time = time.time()
        cpu_usage_before = psutil.cpu_percent(interval=0.1)
        memory_usage_before = psutil.virtual_memory().percent
        
        question = line["text"]
        img_path = os.path.join("/root/autodl-tmp/HealthGPT/neuroblastoma/gallery1500/all", line["image"])

        if img_path:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        else:
            qs = question
        conv = conversation_lib.conv_templates[args.instruct_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + '<start_index>'
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)
        if img_path:
            image = Image.open(img_path).convert('RGB')
            image = expand2square(image, tuple(int(x*255) for x in model.get_vision_tower().image_processor.image_mean))
            image_tensor = model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
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

        response = [int(idx) for idx in re.findall(r'\d+', tokenizer.decode(output_ids[0])[:-8])]
        #print(f'Q: {question}')
        #print(f'HealthGPT: {response}')
        #idx2img(torch.tensor(response).cuda(), args.save_path)
        # 为每个生成的图片创建一个唯一的保存路径
        output_image_path = os.path.join(args.save_path, line["image"])
        # 调用 idx2img 函数并传入新的保存路径
        idx2img(torch.tensor(response).cuda(), output_image_path)
        
        # 在每次推理后记录显存占用、推理时间、CPU和内存使用情况
        end_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # 显存使用量（MB）
        end_time = time.time()
        cpu_usage_after = psutil.cpu_percent(interval=0.1)
        memory_usage_after = psutil.virtual_memory().percent

        # 计算推理时间
        inference_time = end_time - start_time

        # 在循环内部追加写入CSV文件
        output_csv_path = args.occupancy_output_file  # 从参数中获取CSV文件路径
        with open(output_csv_path, 'a', newline='') as csvfile:  # 打开文件以追加模式
            fieldnames = ["Image_Path", "Start_Memory_MB", "End_Memory_MB", "Inference_Time_s", 
                          "CPU_Usage_Before_%", "CPU_Usage_After_%", "Memory_Usage_Before_%", "Memory_Usage_After_%"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 如果文件不存在，写入标题行
            if not os.path.exists(output_csv_path) or os.path.getsize(output_csv_path) == 0:
                writer.writeheader()
            # 写入当前记录
            writer.writerow({
                "Image_Path": line["image"],
                "Start_Memory_MB": start_memory,
                "End_Memory_MB": end_memory,
                "Inference_Time_s": inference_time,
                "CPU_Usage_Before_%": cpu_usage_before,
                "CPU_Usage_After_%": cpu_usage_after,
                "Memory_Usage_Before_%": memory_usage_before,
                "Memory_Usage_After_%": memory_usage_after
            })



if __name__ == "__main__":

    infer()