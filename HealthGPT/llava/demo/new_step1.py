import json
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

class DeepSeekBatchProcessor:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def single_request(self, messages: List[Dict], model: str = "deepseek-reasoner", 
                       max_retries: int = 3, retry_delay: int = 5) -> Dict[str, Any]:
        """单个同步请求，带重试机制"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        for attempt in range(max_retries):
            start_time = time.time()
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
                end_time = time.time()
                
                result = response.json()
                return {
                    "success": True,
                    "request_time": start_time,
                    "response_time": end_time,
                    "duration": end_time - start_time,
                    "input_messages": messages,
                    "response_data": result,
                    "content": result["choices"][0]["message"]["content"] if result.get("choices") and result["choices"][0].get("message") else "",
                    "usage": result.get("usage", {}),
                    "model": result.get("model", model),
                    "attempts": attempt + 1
                }
            except requests.exceptions.Timeout:
                error_msg = f"请求超时 (尝试 {attempt + 1}/{max_retries})"
                print(f"WARN: {error_msg} for messages: {messages[:50]}...") # Print first 50 chars of messages
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    end_time = time.time()
                    return {
                        "success": False,
                        "request_time": start_time,
                        "response_time": end_time,
                        "duration": end_time - start_time,
                        "input_messages": messages,
                        "error": error_msg,
                        "content": "",
                        "usage": {},
                        "model": model,
                        "attempts": attempt + 1
                    }
            except requests.exceptions.RequestException as e:
                error_msg = f"请求失败: {e} (尝试 {attempt + 1}/{max_retries})"
                print(f"WARN: {error_msg} for messages: {messages[:50]}...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    end_time = time.time()
                    return {
                        "success": False,
                        "request_time": start_time,
                        "response_time": end_time,
                        "duration": end_time - start_time,
                        "input_messages": messages,
                        "error": str(e),
                        "content": "",
                        "usage": {},
                        "model": model,
                        "attempts": attempt + 1
                    }
            except Exception as e:
                error_msg = f"发生未知错误: {e} (尝试 {attempt + 1}/{max_retries})"
                print(f"ERROR: {error_msg} for messages: {messages[:50]}...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    end_time = time.time()
                    return {
                        "success": False,
                        "request_time": start_time,
                        "response_time": end_time,
                        "duration": end_time - start_time,
                        "input_messages": messages,
                        "error": str(e),
                        "content": "",
                        "usage": {},
                        "model": model,
                        "attempts": attempt + 1
                    }
        # This part should ideally not be reached if max_retries is > 0
        return {
            "success": False,
            "request_time": time.time(),
            "response_time": time.time(),
            "duration": 0,
            "input_messages": messages,
            "error": "Max retries exceeded without success",
            "content": "",
            "usage": {},
            "model": model,
            "attempts": max_retries
        }
    
    def batch_process(self, message_list: List[List[Dict]], batch_size: int = 5, 
                      max_workers: int = 5) -> List[Dict[str, Any]]:
        """分批处理消息，带失败重试（批次级别）"""
        
        print(f"开始批处理 {len(message_list)} 条消息")
        print(f"批大小: {batch_size}, 最大并发数: {max_workers}")
        
        final_ordered_results = [None] * len(message_list) 

        messages_to_process = [(i, messages) for i, messages in enumerate(message_list)]
        
        retry_round = 0
        while messages_to_process:
            retry_round += 1
            print(f"\n--- 开始处理第 {retry_round} 轮 ---")
            
            current_round_messages = messages_to_process
            messages_to_process = []
            
            total_messages_in_round = len(current_round_messages)
            total_batches_in_round = (total_messages_in_round + batch_size - 1) // batch_size
            
            overall_pbar = tqdm(total=total_messages_in_round, desc=f"轮次 {retry_round} 总体进度", 
                                position=0, leave=True)
            
            for batch_idx_start in range(0, total_messages_in_round, batch_size):
                batch_data = current_round_messages[batch_idx_start:batch_idx_start + batch_size]
                batch_messages_for_request = [item[1] for item in batch_data]
                batch_original_indices = [item[0] for item in batch_data]
                
                batch_start_time = time.time()
                current_batch_num = batch_idx_start // batch_size + 1
                
                print(f"\n处理批次 {current_batch_num}/{total_batches_in_round} (轮次 {retry_round}): "
                      f"消息 {batch_idx_start + 1}-{min(batch_idx_start + len(batch_data), total_messages_in_round)}")
                
                batch_pbar = tqdm(total=len(batch_data), 
                                  desc=f"批次 {current_batch_num} (轮次 {retry_round})", 
                                  position=1, 
                                  leave=False)
                
                temp_batch_results = [None] * len(batch_data)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_local_index = {
                        executor.submit(self.single_request, msg): i
                        for i, msg in enumerate(batch_messages_for_request)
                    }
                    
                    for future in as_completed(future_to_local_index):
                        local_idx = future_to_local_index[future]
                        original_idx = batch_original_indices[local_idx]
                        result = future.result()
                        result["original_message_index"] = original_idx
                        temp_batch_results[local_idx] = result

                        batch_pbar.update(1)
                        status = "✓" if result["success"] else "✗"
                        batch_pbar.set_postfix({"状态": status, "尝试": result.get("attempts", 1)})
                
                batch_pbar.close()
                
                # Process results from this batch
                for i, result in enumerate(temp_batch_results):
                    if result["success"]:
                        final_ordered_results[result["original_message_index"]] = result
                        overall_pbar.update(1)
                    else:
                        messages_to_process.append((result["original_message_index"], result["input_messages"]))
                        overall_pbar.set_postfix({
                            "总成功": sum(1 for r in final_ordered_results if r and r["success"]),
                            "总失败": sum(1 for r in final_ordered_results if r and not r["success"]) + len(messages_to_process)
                        })
                
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                
                successful_in_batch = sum(1 for r in temp_batch_results if r and r["success"])
                failed_in_batch = len(temp_batch_results) - successful_in_batch
                
                print(f"批次完成: 成功 {successful_in_batch}, 失败 {failed_in_batch}, 耗时 {batch_duration:.2f}秒")
                
                if batch_idx_start + batch_size < total_messages_in_round:
                    time.sleep(1)
            
            overall_pbar.close()

            if messages_to_process:
                print(f"\n--- 本轮处理完成，有 {len(messages_to_process)} 条消息需要重试 ---")
                time.sleep(5)

        print("\n所有轮次处理完成。")
        return [res for res in final_ordered_results if res is not None]

def step1_generate_key_features(question: str, answer: str) -> List[Dict]:
    """生成第一步的消息"""
    system_prompt = (
        "You are a helpful assistant that supports Medical Visual Question Answering (VQA) tasks. "
        "Your role is to help identify the key visual elements, anatomical structures, or histological features "
        "that are necessary to answer a given medical visual question. "
        "You should not attempt to answer the question itself—only indicate what regions or features in the image "
        "should be examined in order to answer it correctly."
    )
    user_prompt = (
        "Given the following Medical VQA data, list the visual elements or anatomical structures that must be observed in the image "
        "to answer the question. Do not include any introductory or explanatory text—only list the image features.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Only list the key image features or regions to examine. Do not include any other text."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def load_input_data(file_path: str, num_samples: int = None) -> List[Dict]:
    """加载输入数据，自动处理有meta和没meta的情况，并可以限制加载的样本数量"""
    print(f"正在加载数据文件: {file_path}")
    
    with open(os.path.expanduser(file_path), "r", encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "data" in data:
        print("检测到带meta的数据格式")
        questions = data["data"]
    elif isinstance(data, list):
        print("检测到直接数据数组格式")
        questions = data
    else:
        raise ValueError("不支持的数据格式。期望格式：{'data': [...]} 或 [...]")
    
    if num_samples is not None and num_samples > 0:
        print(f"已限制加载前 {num_samples} 条数据")
        questions = questions[:num_samples]
    
    print(f"成功加载 {len(questions)} 条数据")
    return questions

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/PathLens/playground/pathVQA/train_qa.json")
    # parser.add_argument("--output-file", type=str, default="/root/autodl-tmp/PathLens/playground/pathVQA/step1_results.json")
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/HealthGPT/Cot/MedXpertQA/medxpertqa_min/medxpertqa_min_restructured.json")
    parser.add_argument("--output-file", type=str, default="/root/autodl-tmp/HealthGPT/Cot/MedXpertQA/medxpertqa_min/medxpertqa_min_restults1.json")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=None, help="要处理的样本数量。如果为None，则处理所有样本。")
    
    args = parser.parse_args()
    
    load_dotenv()
    args.api_key = os.getenv("DEEPSEEK_API_KEY")
    if not args.api_key:
        raise ValueError("错误：未找到 DEEPSEEK_API_KEY。请在 .env 文件中或作为环境变量设置它。")
    
    processor = DeepSeekBatchProcessor(args.api_key)

    questions = load_input_data(args.question_file, num_samples=args.num_samples)
    
    print(f"将处理所有 {len(questions)} 个问题")
    
    print("正在准备消息列表...")
    message_list = []
    for line in tqdm(questions, desc="准备消息", total=len(questions)):
        question = line["question"]
        answer = line["answer"]
        messages = step1_generate_key_features(question, answer)
        message_list.append(messages)
    
    print("\n开始并行处理第一步...")
    start_time = time.time()
    
    results = processor.batch_process(
        message_list=message_list,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    end_time = time.time()
    print(f"\n第一步处理完成，总耗时: {end_time - start_time:.2f}秒")
    
    print("正在整理结果...")
    results_map = {res["original_message_index"]: res for res in results}
    
    processed_data = []
    for idx, line in tqdm(enumerate(questions), desc="整理结果", total=len(questions)):
        result = results_map.get(idx)
        if result: 
            processed_item = {
                "original_index": idx,
                "image": line["image"], 
                "question": line["question"],
                "answer": line["answer"],
                "img_path": line["image"],
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/pathvqa", line["image"] + ".jpg"),
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/MedXpertQA/pathology_images11", line["image"]),
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/OmniMedVQA/BreakHis(pathology_images)", line["image"]),
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/OmniMedVQA/(CRC100k)pathology_images", line["image"]),
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/OmniMedVQA/MAlig Lymph（pathology_images）", line["image"]),
                "key_features": result["content"] if result["success"] else "",
                "step1_success": result["success"],
                "step1_error": result.get("error", ""),
                "step1_duration": result["duration"],
                "step1_attempts": result.get("attempts", 1)
            }
        else: # Handle cases where a result might be missing (e.g., if all retries failed for some reason, though the batch_process is designed to prevent this)
            processed_item = {
                "original_index": idx,
                "image": line["image"], 
                "question": line["question"],
                "answer": line["answer"],
                "img_path": line["image"],
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/pathvqa", line["image"] + ".jpg"),
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/MedXpertQA/pathology_images11", line["image"]),
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/OmniMedVQA/BreakHis(pathology_images)", line["image"]),
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/OmniMedVQA/(CRC100k)pathology_images", line["image"]),
                # "img_path": os.path.join("/root/autodl-tmp/database/数据集汇总/images/OmniMedVQA/MAlig Lymph（pathology_images）", line["image"]),
                "key_features": "",
                "step1_success": False,
                "step1_error": "No result obtained after retries",
                "step1_duration": 0,
                "step1_attempts": 0
                
            }
        processed_data.append(processed_item)
    
    success_count = sum(1 for item in processed_data if item["step1_success"])
    failed_count = len(processed_data) - success_count
    
    print("正在保存结果...")
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 处理完成!")
    print(f"结果已保存到: {args.output_file}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"成功率: {success_count/len(processed_data)*100:.1f}%")

if __name__ == "__main__":
    main()