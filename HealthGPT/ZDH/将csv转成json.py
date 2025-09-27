import pandas as pd
import json
import os

def csv_to_vqa_finetune_json_with_full_gpt_value(csv_file_path: str, output_json_path: str):
    """
    将 CSV 文件转换为 vqa_finetune_data_Hov_2945.json 格式的 JSON 文件。
    GPT 的 value 值将从 CSV 中指定的“后处理思维链”那一列完整获取。
    Image_Path 列被视为完整的图片路径，不再进行拼接。

    Args:
        csv_file_path (str): 输入 CSV 文件的路径。
        output_json_path (str): 输出 JSON 文件的路径。
    """
    print(f"正在加载 CSV 文件: {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path)
        print(f"成功加载 {len(df)} 行数据。")
    except FileNotFoundError:
        print(f"错误：文件未找到在 {csv_file_path}")
        return
    except Exception as e:
        print(f"读取 CSV 文件时发生错误: {e}")
        return

    # --- 根据您提供的 CSV 文件的实际列名进行修改 ---
    # CSV 中包含完整图片路径的列名
    IMAGE_FULL_PATH_COLUMN_NAME = "Image_Path" 
    # CSV 中包含问题的列名
    QUESTION_COLUMN_NAME = "问题" 
    # CSV 中包含 GPT 完整输出（即“后处理思维链”）的列名
    GPT_FULL_OUTPUT_COLUMN_NAME = "后处理思维链" 
    # ----------------------------------------------------

    # 检查所需的列是否存在
    required_columns = [IMAGE_FULL_PATH_COLUMN_NAME, QUESTION_COLUMN_NAME, GPT_FULL_OUTPUT_COLUMN_NAME]
    for col in required_columns:
        if col not in df.columns:
            print(f"错误：CSV 文件中缺少必要列 '{col}'。请检查列名并修改脚本。")
            print(f"CSV 文件中的现有列: {df.columns.tolist()}")
            return
    
    json_data = []
    print("正在转换数据...")
    for idx, row in df.iterrows():
        try:
            # 直接从 CSV 获取完整的图片路径
            full_image_path = str(row[IMAGE_FULL_PATH_COLUMN_NAME])
            
            question = str(row[QUESTION_COLUMN_NAME])
            
            # 直接从 CSV 获取 GPT 的完整输出作为 value
            # 检查是否为 NaN 或 None，并转换为字符串
            gpt_full_value = str(row[GPT_FULL_OUTPUT_COLUMN_NAME]) if pd.notna(row[GPT_FULL_OUTPUT_COLUMN_NAME]) else ""
            
            item = {
                "id": str(idx), # 使用行索引作为ID
                "image": full_image_path, # 直接使用完整的图片路径
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": gpt_full_value # 直接使用 CSV 中提取的完整 GPT 输出
                    }
                ]
            }
            json_data.append(item)
        except Exception as e:
            print(f"处理行 {idx} 时发生错误: {e}。该行数据可能不完整或格式不正确。跳过此行。")
            continue

    # 确保输出目录存在
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    print(f"正在保存转换后的 JSON 文件到: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2) # indent=2 使得 JSON 更易读

    print(f"转换完成！总共转换了 {len(json_data)} 条记录。")

if __name__ == "__main__":
    # 定义输入和输出文件路径
    csv_input_file = '/root/autodl-tmp/HealthGPT/ZDH/new/merged_results.csv'
    # 建议命名，以免与原始文件混淆或覆盖
    json_output_file = '/root/autodl-tmp/HealthGPT/ZDH/3000/converted_to_vqa_finetune_format_full_gpt.json' 
    
    # 删除了 image_base_directory 参数，因为它不再需要

    # 调用转换函数
    csv_to_vqa_finetune_json_with_full_gpt_value(csv_input_file, json_output_file)