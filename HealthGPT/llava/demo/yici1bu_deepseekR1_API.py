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
import re

#########################################
#导入API
from openai import OpenAI
client = OpenAI(api_key="sk-3d990190200d42f2b500fe6d8117aa76", base_url="https://api.deepseek.com")
#########################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--dtype', type=str, default='FP32')
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2')
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
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/HealthGPT/VQA-RAD/question1.json")
    parser.add_argument("--output_file", type=str, default="/root/autodl-tmp/HealthGPT/VQA-RAD/question1.csv")
    

    args = parser.parse_args()
    output_file = args.output_file

    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)

    model = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.float16
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
    #确保在模型初始化后，将其移动到GPU上。
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
                           
    #从这里开始要循环遍历实现批量推理
    # 执行 step1
    execute_step1(questions, args.output_file)
    # 执行 step2
    execute_step2(questions, model, tokenizer, args,args.output_file,model_dtype)
    # 执行 step3
    execute_step3(questions, args.output_file)

##################################################################################
# 执行 step1：用 DeepSeek R1 提取图像解读的重点信息
def execute_step1(questions, output_file):
    results = []
    for line in tqdm(questions):
        Q = line["question"]
        A = line["answer"]
        
        #prompt1
        prompt_step1 = (
            "Given the following visual question answering (VQA) data, identify the key visual elements or image features "
            "that are necessary to answer the question. You do not need to answer the question—just list what should be "
            "observed in the image to arrive at the correct answer.\n"
            f"Question: {line['question']}\n"
            f"Answer: {line['answer']}\n"
            "Please list the image features or areas that should be examined to answer this question.")
    
        response_step1 = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt_step1},
            ],
            stream=False)
        
        #提取出回答
        response_step1_text = response_step1.choices[0].message.content
        
        #保存到csv
        results.append({"image": line["image"], "question": Q, "answer": A, "step1": response_step1_text})
        pd.DataFrame(results).to_csv(output_file, mode='w', header=True, index=False, encoding="utf-8-sig")
        
##################################################################################
# 执行 step2：用HealthGPT生成详细描述
def execute_step2(questions, model, tokenizer, args,output_file,model_dtype):
    results = pd.read_csv(output_file)
    for index, row in tqdm(results.iterrows()):
        img_path = os.path.join("/root/autodl-tmp/HealthGPT/VQA-RAD/VQA_RAD Image Folder", row["image"])
        #生成prompt2
        prompt_step2 = (
            "Using the image and the following key visual indicators, generate a detailed medical description of the image. "
            f"Key features to observe: {row['step1']}")
        question = prompt_step2
        
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
        
        response_step2 = tokenizer.decode(output_ids[0], skip_special_tokens=True)[:-8]
        
        results.at[index, 'step2'] = response_step2
        results.to_csv(output_file, mode='w', header=True, index=False, encoding="utf-8-sig")

##################################################################################
# 执行 step3：用deepseekR1得到思维链
def execute_step3(questions, output_file):
    results = pd.read_csv(output_file)
    for index, row in tqdm(results.iterrows()):
        Q = row["question"]
        A = row["answer"]
        prompt_step3 = (
            "You are a medical reasoning assistant. Your task is to analyze the given image description and question, "
            "and generate a detailed, step-by-step clinical reasoning process wrapped inside <think></think> tags. "
            "Then, based on that reasoning, output ONLY the final answer content, without any XML or tags.\n"
            "Note:\n"
            "- The <think> section is only for internal reasoning and should be detailed, logical, and based on the image and question.\n"
            "- At the end, output ONLY the final answer (as text), exactly matching the answer provided in the data.\n"
            "- Do NOT include any <think> or <answer> tags in the final output.\n\n"
            f"Image description: {row['step2']}\n"
            f"Question: {Q}\n"
            f"Ground truth answer: {A}\n"
            "Please begin by thinking step by step, then conclude with the final answer only.\n"
            "Example Output:\n"
            "<think>The question asks whether there is evidence of X. According to the image description, Y is present, which strongly indicates Z. Therefore, it is reasonable to conclude that...</think>\n"
            "Yes\n")
        
        response_step3 = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt_step3},
            ],
            stream=False
        )
        
        #提取思考和答案
        reasoning_content = response_step3.choices[0].message.reasoning_content
        content = response_step3.choices[0].message.content
        
        # 使用正则表达式删除单词 "description"
        reasoning_content = re.sub(r'\bdescription\b', '', reasoning_content)

        #手动加上<>
        output_text = f"<think>{reasoning_content}</think><answer>{content}</answer>"
        
        results.at[index, 'step3'] = output_text
        results.to_csv(output_file, mode='w', header=True, index=False, encoding="utf-8-sig")


if __name__ == "__main__":

    infer()