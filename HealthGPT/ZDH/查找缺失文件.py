import json
import os
import re # 导入正则表达式模块

def find_unconverted_data_strict(train_qa_path, vqa_finetune_path, output_processed_path, output_unconverted_path, limit=3000):
    """
    查找 train_qa.json 中未成功转换到 vqa_finetune_data_Hov_2945.json 的数据项，
    严格根据 image, question, answer 三个字段进行匹配。

    Args:
        train_qa_path (str): train_qa.json 文件的路径。
        vqa_finetune_path (str): vqa_finetune_data_Hov_2945.json 文件的路径。
        output_processed_path (str): 输出处理后的 train_qa 数据的路径。
        output_unconverted_path (str): 输出未转换数据的路径。
        limit (int): 限制处理 train_qa.json 的前多少条数据。
    """
    train_qa_data = []
    vqa_finetune_data = []

    # --- 尝试加载 train_qa.json ---
    print(f"正在尝试加载 {train_qa_path}...")
    try:
        with open(train_qa_path, 'r', encoding='utf-8') as f:
            train_qa_data = json.load(f)
        print(f"成功加载 {len(train_qa_data)} 条数据从 {train_qa_path}。")
    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径是否正确：{train_qa_path}")
        return
    except json.JSONDecodeError as e:
        print(f"错误：解析 {train_qa_path} 失败，文件可能不是有效的 JSON 格式：{e}")
        return
    except Exception as e:
        print(f"加载 {train_qa_path} 时发生未知错误：{e}")
        return

    # --- 尝试加载 vqa_finetune_data_Hov_2945.json ---
    print(f"正在尝试加载 {vqa_finetune_path}...")
    try:
        with open(vqa_finetune_path, 'r', encoding='utf-8') as f:
            vqa_finetune_data = json.load(f)
        print(f"成功加载 {len(vqa_finetune_data)} 条数据从 {vqa_finetune_path}。")
    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径是否正确：{vqa_finetune_path}")
        return
    except json.JSONDecodeError as e:
        print(f"错误：解析 {vqa_finetune_path} 失败，文件可能不是有效的 JSON 格式：{e}")
        return
    except Exception as e:
        print(f"加载 {vqa_finetune_path} 时发生未知错误：{e}")
        return

    # 构建 vqa_finetune_data 的快速查找结构，使用 image, question, answer 的组合
    vqa_lookup = set()
    image_prefix = "/root/autodl-tmp/HealthGPT/playground/data_sft/img/train/"
    image_suffix = ".jpg" # 新增：image 后缀

    print("正在构建 vqa_finetune_data 的查找字典...")
    for i, item in enumerate(vqa_finetune_data):
        try:
            # 修改点：同时移除前缀和后缀
            vqa_image = item.get('image', '')
            if vqa_image.startswith(image_prefix):
                vqa_image = vqa_image[len(image_prefix):]
            if vqa_image.endswith(image_suffix):
                vqa_image = vqa_image[:-len(image_suffix)]
            
            human_question = ""
            gpt_answer = ""

            conversations = item.get('conversations', [])
            for conv in conversations:
                if conv.get('from') == 'human':
                    human_question = conv.get('value', '')
                elif conv.get('from') == 'gpt':
                    # 使用正则表达式提取 <answer></answer> 之间的内容
                    match = re.search(r'<answer>(.*?)</answer>', conv.get('value', ''), re.DOTALL)
                    if match:
                        gpt_answer = match.group(1).strip()
            
            # 确保关键字段不为空，避免生成无效的查找键
            if vqa_image and human_question and gpt_answer:
                lookup_key = f"{vqa_image}___SEP___{human_question}___SEP___{gpt_answer}"
                vqa_lookup.add(lookup_key)
            else:
                print(f"警告：vqa_finetune_data 第 {i} 条数据（ID: {item.get('id', 'N/A')}）的关键字段缺失或为空，跳过构建查找键。")

        except Exception as e:
            print(f"警告：处理 vqa_finetune_data 第 {i} 条数据时发生错误：{e}，跳过此条。")

    print(f"vqa_finetune_data 查找字典构建完成，包含 {len(vqa_lookup)} 个唯一条目。")

    processed_train_qa = []
    unconverted_items = []

    print(f"开始处理 train_qa.json 的前 {min(limit, len(train_qa_data))} 条数据，并进行严格匹配...")
    for i in range(min(limit, len(train_qa_data))):
        qa_item = train_qa_data[i]
        
        # 使用 .get() 方法安全地获取字段，避免 KeyError
        train_image = qa_item.get('image', '')
        train_question = qa_item.get('question', '')
        train_answer = qa_item.get('answer', '')

        # 确保关键字段不为空，这可能是导致匹配失败的原因之一
        if not (train_image and train_question and train_answer):
            print(f"警告：train_qa_data 第 {i} 条数据（原始数据ID可能为：{qa_item.get('id', 'N/A')}）的关键字段缺失 (image, question, answer)，标记为未转换。")
            is_converted = False
        else:
            # 构建 train_qa 对应的查找键
            lookup_key_train = f"{train_image}___SEP___{train_question}___SEP___{train_answer}"
            is_converted = (lookup_key_train in vqa_lookup)

        # 添加到处理后的列表
        processed_item = {
            "id": i,
            "image": train_image,
            "question": train_question,
            "answer": train_answer,
            "converted": is_converted
        }
        processed_train_qa.append(processed_item)

        # 如果未转换，则添加到未转换列表
        if not is_converted:
            unconverted_items.append({
                "id": i,
                "image": train_image,
                "question": train_question,
                "answer": train_answer
            })
            print(f"ID {i}: '{train_image}' - '{train_question}' - '{train_answer}' 未成功转换。")

    print(f"共找到 {len(unconverted_items)} 条未成功转换的数据。")

    # --- 保存处理后的 train_qa 数据 ---
    print(f"正在保存处理后的 train_qa 数据到 {output_processed_path}...")
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_processed_path), exist_ok=True)
        with open(output_processed_path, 'w', encoding='utf-8') as f:
            json.dump(processed_train_qa, f, ensure_ascii=False, indent=4)
        print(f"成功保存到 {output_processed_path}")
    except Exception as e:
        print(f"错误：保存 {output_processed_path} 失败：{e}")

    # --- 保存未转换的数据 ---
    print(f"正在保存未转换数据到 {output_unconverted_path}...")
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_unconverted_path), exist_ok=True)
        with open(output_unconverted_path, 'w', encoding='utf-8') as f:
            json.dump(unconverted_items, f, ensure_ascii=False, indent=4)
        print(f"成功保存到 {output_unconverted_path}")
    except Exception as e:
        print(f"错误：保存 {output_unconverted_path} 失败：{e}")

    print("脚本执行完毕。")

if __name__ == "__main__":
    train_qa_file = "/root/autodl-tmp/HealthGPT/ZDH/train_qa.json"
    vqa_finetune_file = "/root/autodl-tmp/HealthGPT/ZDH/temp/vqa_finetune_data_Hov_2945.json"
    # 修改输出目录，防止覆盖之前的测试结果，并确保目录存在
    output_processed_train_qa = "/root/autodl-tmp/HealthGPT/ZDH/new/train_qa_processed_strict.json"
    output_unconverted = "/root/autodl-tmp/HealthGPT/ZDH/new/unconverted_data_strict.json"

    find_unconverted_data_strict(train_qa_file, vqa_finetune_file, output_processed_train_qa, output_unconverted)