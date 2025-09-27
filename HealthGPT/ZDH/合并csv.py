import pandas as pd
import json
import os

def merge_csv_based_on_json(
    json_path,
    csv_true_path,
    csv_false_path,
    output_csv_path,
    image_base_path="/root/autodl-tmp/HealthGPT/playground/data_sft/img/train/"
):
    """
    根据 JSON 文件中的 converted 字段，从两个 CSV 文件中查找并合并数据。

    Args:
        json_path (str): train_qa_processed_strict.json 文件的路径。
        csv_true_path (str): converted=true 时查找的 CSV 文件路径。
        csv_false_path (str): converted=false 时查找的 CSV 文件路径。
        output_csv_path (str): 合并后的 CSV 文件输出路径。
        image_base_path (str): 图片路径的前缀，用于构造完整的图片路径。
    """
    print(f"正在加载 JSON 文件: {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        print(f"成功加载 {len(json_data)} 条 JSON 数据。")
    except FileNotFoundError:
        print(f"错误：JSON 文件未找到，请检查路径是否正确：{json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"错误：解析 JSON 文件失败，文件可能不是有效的 JSON 格式：{e}")
        return
    except Exception as e:
        print(f"加载 JSON 文件时发生未知错误：{e}")
        return

    print(f"正在加载 converted=True 对应的 CSV 文件: {csv_true_path}...")
    try:
        df_true = pd.read_csv(csv_true_path)
        print(f"成功加载 {len(df_true)} 条数据从 {csv_true_path}。")
    except FileNotFoundError:
        print(f"错误：CSV 文件未找到，请检查路径是否正确：{csv_true_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"警告：{csv_true_path} 是空文件，跳过加载。")
        df_true = pd.DataFrame(columns=[
            "Image_Path", "问题", "答案", "图像特征", "图像描述",
            "初始思维链", "初始思维链翻译", "是否需要后处理", "后处理思维链",
            "后处理思维链翻译", "处理状态", "错误信息"
        ])
    except Exception as e:
        print(f"加载 {csv_true_path} 时发生未知错误：{e}")
        return

    print(f"正在加载 converted=False 对应的 CSV 文件: {csv_false_path}...")
    try:
        df_false = pd.read_csv(csv_false_path)
        print(f"成功加载 {len(df_false)} 条数据从 {csv_false_path}。")
    except FileNotFoundError:
        print(f"错误：CSV 文件未找到，请检查路径是否正确：{csv_false_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"警告：{csv_false_path} 是空文件，跳过加载。")
        df_false = pd.DataFrame(columns=[
            "Image_Path", "问题", "答案", "图像特征", "图像描述",
            "初始思维链", "初始思维链翻译", "是否需要后处理", "后处理思维链",
            "后处理思维链翻译", "处理状态", "错误信息"
        ])
    except Exception as e:
        print(f"加载 {csv_false_path} 时发生未知错误：{e}")
        return

    # 创建一个用于快速查找的字典，键是 (Image_Path, 问题, 答案) 的组合
    # 假设 CSV 中的 Image_Path 已经是完整的路径
    lookup_true = {}
    if not df_true.empty:
        for _, row in df_true.iterrows():
            key = (row["Image_Path"], row["问题"], row["答案"])
            lookup_true[key] = row.to_dict()

    lookup_false = {}
    if not df_false.empty:
        for _, row in df_false.iterrows():
            key = (row["Image_Path"], row["问题"], row["答案"])
            lookup_false[key] = row.to_dict()

    merged_records = []
    found_count = 0
    not_found_count = 0

    print("开始根据 JSON 数据进行合并...")
    for i, item in enumerate(json_data):
        image_name = item.get('image', '')
        question = item.get('question', '')
        answer = item.get('answer', '')
        converted = item.get('converted', False)
        
        # 构造 Image_Path, 这是 CSV 中对应的完整路径
        full_image_path = os.path.join(image_base_path, f"{image_name}.jpg")

        lookup_key = (full_image_path, question, answer)

        record_to_add = None

        if converted:
            if lookup_key in lookup_true:
                record_to_add = lookup_true[lookup_key]
                found_count += 1
            else:
                print(f"警告：JSON ID {i} (converted=True) 未在 '{csv_true_path}' 中找到匹配项: {lookup_key}")
                not_found_count += 1
        else: # converted is false
            if lookup_key in lookup_false:
                record_to_add = lookup_false[lookup_key]
                found_count += 1
            else:
                print(f"警告：JSON ID {i} (converted=False) 未在 '{csv_false_path}' 中找到匹配项: {lookup_key}")
                not_found_count += 1
        
        if record_to_add:
            merged_records.append(record_to_add)
        else:
            # 如果没有找到匹配项，添加一个空行或者只包含已知信息的行
            # 确保列名与原始CSV文件一致，避免写入时报错
            empty_record = {
                "Image_Path": full_image_path,
                "问题": question,
                "答案": answer,
                "图像特征": "",
                "图像描述": "",
                "初始思维链": "",
                "初始思维链翻译": "",
                "是否需要后处理": "",
                "后处理思维链": "",
                "后处理思维链翻译": "",
                "处理状态": "未找到匹配数据",
                "错误信息": f"JSON中converted为{converted}，但在对应CSV中未找到匹配项。"
            }
            merged_records.append(empty_record)

    if not merged_records:
        print("没有找到任何可合并的记录，输出文件将为空或只包含列头。")
        # 确保即使没有找到记录也能创建带有正确列名的DataFrame
        final_df = pd.DataFrame(columns=[
            "Image_Path", "问题", "答案", "图像特征", "图像描述",
            "初始思维链", "初始思维链翻译", "是否需要后处理", "后处理思维链",
            "后处理思维链翻译", "处理状态", "错误信息"
        ])
    else:
        final_df = pd.DataFrame(merged_records)

    print(f"总计找到 {found_count} 条匹配记录。")
    print(f"总计 {not_found_count} 条记录未在对应 CSV 中找到匹配。")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在保存合并后的 CSV 文件到: {output_csv_path}...")
    try:
        final_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"合并完成！文件已保存到 {output_csv_path}")
    except Exception as e:
        print(f"错误：保存合并后的 CSV 文件失败：{e}")


if __name__ == "__main__":
    json_file = "/root/autodl-tmp/HealthGPT/ZDH/new/train_qa_processed_strict.json"
    csv_true_file = "/root/autodl-tmp/HealthGPT/ZDH/temp/step3_results_new.csv"
    csv_false_file = "/root/autodl-tmp/HealthGPT/ZDH/another/step3_results_fixed_image_path.csv"
    output_merged_csv = "/root/autodl-tmp/HealthGPT/ZDH/new/merged_results.csv"

    merge_csv_based_on_json(json_file, csv_true_file, csv_false_file, output_merged_csv)