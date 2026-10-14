import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# 1. 设置包含您所有CSV文件的文件夹路径
# 请将下面的路径替换为您的实际文件夹路径
folder_path = './'

# 2. 自动查找文件夹内所有的csv文件
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

if not csv_files:
    print(f"错误：在文件夹 '{folder_path}' 中没有找到任何CSV文件。请检查路径。")
else:
    all_data = []
    # 3. 循环读取和处理每个文件
    for file_path in csv_files:
        # 从文件名中提取 goal_xy 参数
        filename = os.path.basename(file_path)
        goal_xy = 'unknown'
        if "0.1_TQC_1.csv" in filename:
            goal_xy = 0.1
        elif "0.3.csv" in filename:
            goal_xy = 0.3
        elif "0.5_TQC_1.csv" in filename:
            goal_xy = 0.5
        else:
            continue

        # 读取CSV数据
        df = pd.read_csv(file_path)
        df['goal_xy'] = goal_xy
        all_data.append(df)

    # 4. 合并所有数据并进行清洗
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['Step'] = pd.to_numeric(combined_df['Step'], errors='coerce')
    # Tensorboard的Value通常是0-1之间，转换为百分比
    combined_df['Success Rate'] = pd.to_numeric(combined_df['Value'], errors='coerce') * 100
    combined_df.dropna(subset=['Step', 'Success Rate'], inplace=True)

    # 5. 使用Seaborn绘图
    # 设置风格
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 8))

    # 绘制线图，自动计算均值和标准差
    sns.lineplot(data=combined_df, x='Step', y='Success Rate', hue='goal_xy', palette='viridis', errorbar='sd')

    # 6. 美化图表
    plt.title('Learning Curve Comparison for Different Goal Space Sizes', fontsize=16)
    plt.xlabel('Training Timesteps', fontsize=12)
    plt.ylabel('Mean Success Rate (%)', fontsize=12)
    plt.legend(title='Training Goal Space (goal_xy)')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # X轴使用科学计数法
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 7. 显示或保存图表
    plt.show()
    # 如果您想保存图片，可以取消下面这行代码的注释
    # plt.savefig('goal_space_comparison.png', dpi=300)