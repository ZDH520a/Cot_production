import pickle
import json
import os

def pkl_to_json(pkl_filepath, json_filepath):
    """
    将.pkl文件转换为.json文件。

    Args:
        pkl_filepath (str): .pkl文件的完整路径。
        json_filepath (str): 输出.json文件的完整路径。
    """
    try:
        # 检查.pkl文件是否存在
        if not os.path.exists(pkl_filepath):
            print(f"错误：'{pkl_filepath}' 文件不存在。")
            return

        # 加载.pkl文件
        with open(pkl_filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 将数据写入.json文件
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"成功将 '{pkl_filepath}' 转换为 '{json_filepath}'")

    except Exception as e:
        print(f"转换过程中发生错误：{e}")

if __name__ == "__main__":
    # 定义输入和输出文件路径
    pkl_file = "/root/autodl-tmp/HealthGPT/ZDH/train_qa.pkl"
    json_file = "/root/autodl-tmp/HealthGPT/ZDH/train_qa.json" # 你可以修改这个输出文件名和路径

    # 调用函数进行转换
    pkl_to_json(pkl_file, json_file)