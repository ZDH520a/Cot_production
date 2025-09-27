import json

def update_original_index():
    # 读取两个JSON文件
    step2_path = "/root/autodl-tmp/HealthGPT/ZDH/another/step2_results.json"
    unconverted_path = "/root/autodl-tmp/HealthGPT/ZDH/new/unconverted_data_strict.json"
    output_path = "/root/autodl-tmp/HealthGPT/ZDH/another/updated_step2_results.json"
    
    # 读取step2_results.json
    with open(step2_path, 'r', encoding='utf-8') as f:
        step2_data = json.load(f)
    
    # 读取unconverted_data_strict.json
    with open(unconverted_path, 'r', encoding='utf-8') as f:
        unconverted_data = json.load(f)
    
    # 验证数据长度是否一致
    if len(step2_data) != len(unconverted_data):
        print(f"警告：数据长度不一致！step2_results.json有{len(step2_data)}个项目，unconverted_data_strict.json有{len(unconverted_data)}个项目")
        return
    
    print(f"开始处理{len(step2_data)}个项目...")
    
    # 更新original_index字段
    for i in range(len(step2_data)):
        # 获取unconverted_data中对应位置的id
        new_original_index = unconverted_data[i]['id']
        
        # 更新step2_data中的original_index
        step2_data[i]['original_index'] = new_original_index
        
        # 打印前几个更新的信息用于验证
        if i < 5:
            print(f"项目{i}: original_index更新为{new_original_index}, image: {step2_data[i]['image']}")
    
    # 保存更新后的数据到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(step2_data, f, indent=2, ensure_ascii=False)
    
    print(f"更新完成！结果已保存到: {output_path}")
    print(f"总共更新了{len(step2_data)}个项目的original_index字段")

if __name__ == "__main__":
    update_original_index()