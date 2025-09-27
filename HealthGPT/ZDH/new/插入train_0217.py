import pandas as pd
import os

# 读取两个CSV文件
file1_path = "/root/autodl-tmp/HealthGPT/ZDH/train_0217/step3_results.csv"
file2_path = "/root/autodl-tmp/HealthGPT/ZDH/another/step3_results_filled_image_path.csv"

try:
    # 读取第一个文件（只有一行数据）
    df1 = pd.read_csv(file1_path)
    print(f"文件1读取成功，共有 {len(df1)} 行数据")
    print(f"文件1的列名: {list(df1.columns)}")
    
    # 读取第二个文件
    df2 = pd.read_csv(file2_path)
    print(f"文件2读取成功，共有 {len(df2)} 行数据")
    print(f"文件2的列名: {list(df2.columns)}")
    
    # 检查列名是否一致
    if list(df1.columns) != list(df2.columns):
        print("警告：两个文件的列名不完全一致")
        print(f"文件1独有的列: {set(df1.columns) - set(df2.columns)}")
        print(f"文件2独有的列: {set(df2.columns) - set(df1.columns)}")
    
    # 获取要插入的行数据
    row_to_insert = df1.iloc[0].copy()
    
    # 修改Image_Path列的值
    if 'Image_Path' in row_to_insert.index:
        row_to_insert['Image_Path'] = "/root/autodl-tmp/HealthGPT/playground/data_sft/img/train/train_0217.jpg"
        print("已更新Image_Path列的值")
    else:
        print("警告：未找到Image_Path列")
    
    # 插入位置（第340行，从0开始计数）
    insert_position = 340
    
    # 检查插入位置是否有效
    if insert_position > len(df2):
        print(f"警告：插入位置 {insert_position} 超出了文件2的行数 {len(df2)}，将插入到末尾")
        insert_position = len(df2)
    
    # 创建新的DataFrame
    # 将df2分割成插入位置前后两部分
    df_before = df2.iloc[:insert_position]
    df_after = df2.iloc[insert_position:]
    
    # 创建包含要插入行的DataFrame
    row_df = pd.DataFrame([row_to_insert])
    
    # 合并三部分
    result_df = pd.concat([df_before, row_df, df_after], ignore_index=True)
    
    print(f"处理完成！新文件共有 {len(result_df)} 行数据")
    print(f"原文件2有 {len(df2)} 行，插入1行后变为 {len(result_df)} 行")
    
    # 保存结果文件
    output_path = "/root/autodl-tmp/HealthGPT/ZDH/another/step3_results_filled_image_path_updated.csv"
    result_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")
    
    # 显示插入位置附近的几行数据用于验证
    print("\n插入位置附近的数据预览：")
    start_idx = max(0, insert_position - 2)
    end_idx = min(len(result_df), insert_position + 3)
    preview_df = result_df.iloc[start_idx:end_idx]
    
    # 只显示前几列以便查看
    cols_to_show = list(result_df.columns)[:5]  # 显示前5列
    if 'Image_Path' in result_df.columns and 'Image_Path' not in cols_to_show:
        cols_to_show.append('Image_Path')
    
    print(preview_df[cols_to_show].to_string(index=True))
    
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except Exception as e:
    print(f"处理过程中出现错误: {e}")