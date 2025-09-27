import json
import os

def find_extra_entries_in_processed_file(train_qa_path, processed_vqa_path, num_train_qa_to_consider=3000):
    """
    查找 processed_vqa_path 文件中存在，但在 train_qa_path 文件的前
    num_train_qa_to_consider 条记录中不存在的样例。

    Args:
        train_qa_path (str): 原始 train_qa.json 文件的路径。
        processed_vqa_path (str): 已处理的 vqa_finetune_data_Hov_2945.json 文件的路径。
        num_train_qa_to_consider (int): 从 train_qa.json 文件中考虑的初始记录数量。

    Returns:
        list: 包含在 processed_vqa_path 中存在但在 train_qa_path 前N条中不存在的记录的列表。
              返回的格式将是 processed_vqa.json 的格式。
    """
    with open(train_qa_path, 'r', encoding='utf-8') as f:
        train_qa_data = json.load(f)

    with open(processed_vqa_path, 'r', encoding='utf-8') as f:
        processed_vqa_data = json.load(f)

    # 从 train_qa_data 的前 num_train_qa_to_consider 条记录中创建 (图片ID, 问题) 集合
    train_qa_subset_set = set()
    # 确保只遍历前 num_train_qa_to_consider 条
    for entry in train_qa_data[:num_train_qa_to_consider]:
        image_id = entry['image']
        question = entry['question']
        train_qa_subset_set.add((image_id, question))

    extra_entries = []
    for entry_processed in processed_vqa_data:
        image_full_path = entry_processed['image']
        # 提取文件名（例如：'train_0422.jpg'）
        image_filename = os.path.basename(image_full_path)
        # 移除 '.jpg' 扩展名，得到 'train_0422' 这种形式的图片ID
        image_id = image_filename.replace('.jpg', '')
        
        question = ""
        for conversation in entry_processed['conversations']:
            if conversation['from'] == 'human':
                question = conversation['value']
                break
        
        # 只有在成功提取到问题后才进行比较
        if question and (image_id, question) not in train_qa_subset_set:
            extra_entries.append(entry_processed)

    return extra_entries

if __name__ == "__main__":
    train_qa_file = '/root/autodl-tmp/HealthGPT/ZDH/train_qa.json'
    processed_vqa_file = '/root/autodl-tmp/HealthGPT/ZDH/temp/vqa_finetune_data_Hov_2945.json'
    
    # 查找 processed_vqa_file 中存在但 train_qa_file 前 3000 条中不存在的记录
    extra_items = find_extra_entries_in_processed_file(train_qa_file, processed_vqa_file, num_train_qa_to_consider=3000)

    print(f"在 {processed_vqa_file} 中找到了 {len(extra_items)} 条记录，但它们不在 {train_qa_file} 的前 3000 条记录中。")
    
    # 将这些“额外”记录保存到一个新的 JSON 文件中
    output_file_path = '/root/autodl-tmp/HealthGPT/ZDH/extra_in_vqa_finetune_data.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(extra_items, f, ensure_ascii=False, indent=4)
    print(f"这些额外记录已保存到: {output_file_path}")

    # 打印一些额外记录的示例进行验证
    if extra_items:
        print("\n以下是一些额外记录的示例:")
        for i, entry in enumerate(extra_items[:5]): # 最多打印 5 个示例
            print(json.dumps(entry, indent=4, ensure_ascii=False))
            if i == 4:
                break