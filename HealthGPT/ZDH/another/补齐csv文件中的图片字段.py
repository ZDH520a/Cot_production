import pandas as pd
import json
import os

def correct_image_paths_and_verify(
    csv_path="/root/autodl-tmp/HealthGPT/ZDH/another/step3_results_filled_image_path_updated.csv",
    json_path="/root/autodl-tmp/HealthGPT/ZDH/another/updated_step2_results.json",
    output_csv_path="/root/autodl-tmp/HealthGPT/ZDH/another/step3_results_fixed_image_path.csv"
):
    """
    根据 JSON 文件的顺序，修正 CSV 文件中的 'Image_Path' 字段，
    并验证 '问题'/'question' 和 '答案'/'answer' 字段的一致性。

    Args:
        csv_path (str): 待修改的 CSV 文件路径。
        json_path (str): 提供正确 'img_path' 和用于验证的 JSON 文件路径。
        output_csv_path (str): 修改后 CSV 文件的输出路径。
    """
    print(f"正在加载 CSV 文件: {csv_path}...")
    try:
        df_csv = pd.read_csv(csv_path)
        print(f"成功加载 {len(df_csv)} 条数据从 {csv_path}。")
    except FileNotFoundError:
        print(f"错误：CSV 文件未找到，请检查路径是否正确：{csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"警告：{csv_path} 是空文件，无法进行修正和验证。")
        return
    except Exception as e:
        print(f"加载 {csv_path} 时发生未知错误：{e}")
        return

    print(f"正在加载 JSON 文件: {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        print(f"成功加载 {len(json_data)} 条 JSON 数据从 {json_path}。")
    except FileNotFoundError:
        print(f"错误：JSON 文件未找到，请检查路径是否正确：{json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"错误：解析 JSON 文件失败，文件可能不是有效的 JSON 格式：{e}")
        return
    except Exception as e:
        print(f"加载 JSON 文件时发生未知错误：{e}")
        return

    if len(df_csv) != len(json_data):
        print(f"警告：CSV 文件 ({len(df_csv)} 条) 和 JSON 文件 ({len(json_data)} 条) 的记录数量不一致。这将影响按顺序匹配的准确性。")

    # 创建一个新的列表来存储修改后的行
    modified_rows = []
    
    mismatched_questions = 0
    mismatched_answers = 0
    total_records_processed = min(len(df_csv), len(json_data))

    print("开始修正 Image_Path 并验证数据一致性...")
    for i in range(total_records_processed):
        csv_row = df_csv.iloc[i].copy() # 获取CSV行并复制，以便修改
        json_item = json_data[i]

        # 1. 修正 Image_Path
        expected_image_path = json_item.get("img_path", "")
        if not expected_image_path:
            print(f"警告：JSON ID {i} 缺少 'img_path' 字段，CSV中对应的 'Image_Path' 将保持不变。")
        else:
            if csv_row["Image_Path"] != expected_image_path:
                print(f"修正 Image_Path: CSV ID {i} 从 '{csv_row['Image_Path']}' 修改为 '{expected_image_path}'")
                csv_row["Image_Path"] = expected_image_path
            # else:
            #     print(f"CSV ID {i} 的 Image_Path 已经正确。") # 如果需要详细日志可以打开

        # 2. 验证 问题/question
        csv_question = csv_row.get("问题", "")
        json_question = json_item.get("question", "")
        if csv_question != json_question:
            mismatched_questions += 1
            print(f"不一致警告：CSV ID {i} 的 '问题' ('{csv_question}') 与 JSON 的 'question' ('{json_question}') 不匹配。")

        # 3. 验证 答案/answer
        csv_answer = csv_row.get("答案", "")
        json_answer = json_item.get("answer", "")
        if csv_answer != json_answer:
            mismatched_answers += 1
            print(f"不一致警告：CSV ID {i} 的 '答案' ('{csv_answer}') 与 JSON 的 'answer' ('{json_answer}') 不匹配。")
            
        modified_rows.append(csv_row)

    # 将修改后的行转换为新的DataFrame
    df_modified_csv = pd.DataFrame(modified_rows, columns=df_csv.columns) # 保持原有列顺序和名称

    print(f"\n修正和验证完成。")
    print(f"总计处理了 {total_records_processed} 条记录。")
    print(f"发现 {mismatched_questions} 处 '问题' 字段不一致。")
    print(f"发现 {mismatched_answers} 处 '答案' 字段不一致。")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在保存修正后的 CSV 文件到: {output_csv_path}...")
    try:
        df_modified_csv.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"修正后的 CSV 文件已成功保存到 {output_csv_path}")
    except Exception as e:
        print(f"错误：保存修正后的 CSV 文件失败：{e}")

if __name__ == "__main__":
    correct_image_paths_and_verify()