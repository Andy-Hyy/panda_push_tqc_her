import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 配置区 ---
# CSV源文件路径
CSV_FILE_PATH = 'batch_evaluation_results_final.csv'
# 生成的图表文件名
OUTPUT_IMAGE_PATH = 'fixed_models_generalization_comparison.png'
# 选择要对比的四种训练类型
TRAIN_TYPES_TO_PLOT = ['Fixed_0.1', 'Fixed_0.3', 'Fixed_0.5', 'Fixed_0.8']

def plot_fixed_model_generalization():
    """
    读取CSV数据，处理并绘制指定模型的空间泛化能力对比图。
    """
    # --- 图表样式设置 ---
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica'] # 使用通用字体以避免乱码
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 300


    # 1. 读取并预处理数据
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"错误：找不到CSV文件 '{CSV_FILE_PATH}'。请确保文件路径正确。")
        return

    # 2. 筛选出需要绘制的数据
    # 确保'train_type'列是字符串类型，以便正确筛选
    df['train_type'] = df['train_type'].astype(str)
    df_filtered = df[df['train_type'].isin(TRAIN_TYPES_TO_PLOT)].copy()
    
    if df_filtered.empty:
        print(f"错误：在CSV文件中没有找到任何指定的数据类型：{TRAIN_TYPES_TO_PLOT}")
        print(f"文件中可用的 train_type 包括: {df['train_type'].unique()}")
        return

    # 确保数据类型正确以便计算
    df_filtered['test_goal_xy'] = pd.to_numeric(df_filtered['test_goal_xy'])
    df_filtered['success_rate'] = pd.to_numeric(df_filtered['success_rate'])

    # 3. 按训练类型和测试目标范围分组，计算成功率的均值和标准差
    #    这会自动合并所有 train_seed 和 eval_seed 的结果
    summary = df_filtered.groupby(['train_type', 'test_goal_xy'])['success_rate'].agg(['mean', 'std']).reset_index()

    # 4. 绘图
    plt.figure(figsize=(14, 9))
    
    # 为不同的训练类型定义外观
    style_map = {
        'Fixed_0.1': {'color': '#1f77b4', 'linestyle': '-', 'label': 'Trained on 0.1'},   # 蓝色
        'Fixed_0.3': {'color': '#2ca02c', 'linestyle': '--', 'label': 'Trained on 0.3'},  # 绿色
        'Fixed_0.5': {'color': '#ff7f0e', 'linestyle': ':', 'label': 'Trained on 0.5'},   # 橙色
        'Fixed_0.8': {'color': '#d62728', 'linestyle': '-.', 'label': 'Trained on 0.8'}   # 红色
    }
    
    # 手动排序，让图例顺序更合理
    sorted_train_types = sorted(summary['train_type'].unique(), key=lambda x: float(x.split('_')[-1]))

    for train_type in sorted_train_types:
        group_df = summary[summary['train_type'] == train_type]
        style = style_map.get(train_type)
        if not style:
            continue

        # 绘制均值线
        plt.plot(group_df['test_goal_xy'], group_df['mean'], 
                 label=style['label'], 
                 color=style['color'],
                 linestyle=style['linestyle'],
                 linewidth=2.5,
                 marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2)
        
        # 绘制标准差范围（阴影）
        plt.fill_between(group_df['test_goal_xy'], 
                         group_df['mean'] - group_df['std'], 
                         group_df['mean'] + group_df['std'], 
                         color=style['color'],
                         alpha=0.15)
        
        # 添加一条垂直虚线，标注该模型的训练范围
        try:
            training_value = float(train_type.split('_')[-1])
            plt.axvline(x=training_value, color=style['color'], linestyle='--', linewidth=1.5, alpha=0.7)
        except (ValueError, IndexError):
            print(f"无法为 {train_type} 解析训练值以绘制垂直线。")

    # 5. 格式化图表
    plt.title('Generalization Performance of Models Trained on Fixed Goal Ranges', fontsize=18, pad=20)
    plt.xlabel('Evaluation Goal Range (Generalization Test)', fontsize=14)
    plt.ylabel('Mean Success Rate', fontsize=14)
    
    # 创建一个虚拟的线来为axvline添加图例
    plt.plot([], [], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Training Point')
    
    plt.legend(title='Model Type', fontsize=12, title_fontsize=13, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.xticks(np.arange(0.1, 0.9, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0.05, 0.85)
    plt.ylim(-0.05, 1.05)
    
    # 调整布局以适应图例
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # 6. 保存图表
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"图表已成功保存至: {OUTPUT_IMAGE_PATH}")


if __name__ == '__main__':
    plot_fixed_model_generalization()