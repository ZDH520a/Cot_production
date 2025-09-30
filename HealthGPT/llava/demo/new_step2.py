import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import json
import time
import torch
import transformers
import tokenizers
from tqdm import tqdm
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
from llava.peft import LoraConfig, get_peft_model
from PIL import Image
import argparse
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utils
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_weights(model, hlora_path, fusion_layer_path=None):
    hlora_weights = torch.load(hlora_path)
    hlora_unexpected_keys = model.load_state_dict(hlora_weights, strict=False)[1]
    if hlora_unexpected_keys:
        print(f"Warning: Unexpected keys in hlora checkpoint: {hlora_unexpected_keys}")
    return model

com_vision_args = argparse.Namespace(
    freeze_backbone=False,
    mm_patch_merge_type='flat',
    mm_projector_type='mlp2x_gelu',
    mm_use_im_patch_token=False,
    mm_use_im_start_end=False,
    mm_vision_select_feature='patch',
    mm_vision_select_layer=-2,
    model_name_or_path=None,
    pretrain_mm_mlp_adapter=None,
    tune_mm_mlp_adapter=False,
    version=None,
    vision_tower=None
)

def load_multimodal_model(args):
    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)
    print(f"使用的数据类型: {model_dtype}")
    print(f"使用的注意力实现: {args.attn_implementation}")

    model = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=model_dtype
    ).to(device)

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
    parser.add_argument('--dtype', type=str, default='FP16', choices=['FP32', 'FP16', 'BF16'])
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2')
    parser.add_argument('--hlora_r', type=int, default=64)
    parser.add_argument('--hlora_alpha', type=int, default=128)
    parser.add_argument('--hlora_dropout', type=float, default=0.0)
    parser.add_argument('--hlora_nums', type=int, default=4)
    parser.add_argument('--instruct_template', type=str, default='phi3_instruct')
    parser.add_argument('--vit_path', type=str, default="/root/autodl-tmp/HealthGPT/models/clip-vit-large-patch14-336/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
    parser.add_argument('--hlora_path', type=str, default="/root/autodl-tmp/HealthGPT/models/HealthGPT-M3/models--lintw--HealthGPT-M3/snapshots/a1184cc40b3341509e00549a2b19d2c1c8ff955c/com_hlora_weights.bin")
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument("--step1-file", type=str, default="/root/autodl-tmp/HealthGPT/Cot/MedXpertQA/medxpertqa_min/medxpertqa_min_restults1.json")
    parser.add_argument("--output-file", type=str, default="/root/autodl-tmp/HealthGPT/Cot/MedXpertQA/medxpertqa_min/medxpertqa_min_restults2.json")
    
    args = parser.parse_args()
    
    if args.attn_implementation == 'flash_attention_2' and args.dtype == 'FP32':
        print("警告: FlashAttention只支持fp16和bf16数据类型，自动将dtype从FP32修改为FP16以保证兼容性")
        args.dtype = 'FP16'

    model, tokenizer = load_multimodal_model(args)
    print("HealthGPT 模型加载完成")
    
    print(f"正在加载第一步结果: {args.step1_file}")
    with open(args.step1_file, 'r', encoding='utf-8') as f:
        all_step1_data = json.load(f)
    
    print(f"加载了 {len(all_step1_data)} 条第一步的结果")
    
    # 筛选出step1_success为True的数据
    step1_success_data = [item for item in all_step1_data if item.get("step1_success", False) == True]
    step1_failed_data = [item for item in all_step1_data if item.get("step1_success", False) != True]
    
    print(f"第一步成功的数据: {len(step1_success_data)} 条")
    print(f"第一步失败的数据: {len(step1_failed_data)} 条")

    print("开始处理第二步...")
    start_time = time.time()   
    processed_results = []
    
    # 处理第一步成功的数据
    for idx, item in enumerate(tqdm(step1_success_data, desc="处理第二步(成功项)")):
        try:
            step2_prompt = concat_picture_future(item["question"], item["key_features"])
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
    step2_processed_count = len(step1_success_data)
    
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