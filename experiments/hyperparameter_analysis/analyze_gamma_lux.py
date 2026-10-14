import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- 目录与文件模式定义 -----------
# 脚本默认在'logs'目录的父目录中运行
CSV_DIR = "logs/FrankaPushSparse-v0_TQC_gamma_test/Tables"
gamma90_pattern = "gamma0.90_seed*.csv"
gamma95_pattern = "gamma0.95_seed*.csv"
gamma99_pattern = "gamma0.99_seed*.csv"
TABLE_OUTPUT_PATH = "gamma_performance_summary.csv"
PLOT_OUTPUT_PATH = "LearningCurve_Gamma_Comparison_Styled.png"

# ----------- 辅助函数 -----------

def load_group(csv_pattern):
    """根据模式加载所有CSV文件，返回对齐的步数和成功率数组"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(base_dir, csv_pattern)
    csv_files = sorted(glob.glob(search_path))
    if not csv_files:
        print(f"警告: 未在 '{search_path}' 中找到匹配 '{os.path.basename(csv_pattern)}' 的CSV文件")
        return []
    print(f"加载了 {len(csv_files)} 个运行: {[os.path.basename(f) for f in csv_files]}")
    runs = []
    for file in csv_files:
        df = pd.read_csv(file)
        step_col_list = [c for c in df.columns if "step" in c.lower()]
        step_col = step_col_list[0] if step_col_list else df.columns[0]
        sr_col_list = [c for c in df.columns if "success" in c.lower() or "value" in c.lower()]
        sr_col = sr_col_list[0] if sr_col_list else df.columns[-1]
        print(f"  > 文件: {os.path.basename(file):<25} | 步数-列: '{step_col}' | 成功率-列: '{sr_col}'")
        runs.append((df[step_col].to_numpy(), df[sr_col].to_numpy()))
    return runs

def align_curves(runs, num_points=2000):
    """将所有曲线插值到一个公共的x轴上 (便于计算平均值)"""
    if not runs:
        return np.array([]), np.array([])
    max_step = min(r[0].max() for r in runs)
    min_step = max(r[0].min() for r in runs)
    common_x = np.linspace(min_step, max_step, num_points)
    aligned_y = []
    for x, y in runs:
        aligned = np.interp(common_x, x, y)
        aligned_y.append(aligned)
    return common_x, np.vstack(aligned_y)

def plot_learning_curve(x, y_mat, label):
    """
    绘制平均学习曲线及其标准差范围
    (已修复颜色同步问题)
    """
    mean_y = np.mean(y_mat, axis=0)
    std_y = np.std(y_mat, axis=0)
    
    # 1. 绘制平均值曲线，并获取返回的Line2D对象
    line = plt.plot(x, mean_y, label=label, linewidth=2)[0]

    # 2. 从Line2D对象中获取刚绘制的曲线的实际颜色
    line_color = line.get_color()

    # 3. 使用捕获到的颜色来填充置信区间，确保颜色一致
    plt.fill_between(x, np.clip(mean_y - std_y, 0, 1), np.clip(mean_y + std_y, 0, 1), 
                     color=line_color, alpha=0.25)

def calc_sample_efficiency(x, mean_y, threshold=0.8):
    """计算达到指定成功率阈值所需的样本数（步数）"""
    idx = np.where(mean_y >= threshold)[0]
    return int(x[idx[0]]) if len(idx) > 0 else np.nan

def calc_asymptotic(y_mat, portion=0.2):
    """计算渐进性能 (训练最后20%部分的平均成功率)"""
    n = y_mat.shape[1]
    last_n = int(n * portion)
    asymp_mean_per_run = np.mean(y_mat[:, -last_n:], axis=1)
    return np.mean(asymp_mean_per_run), np.std(asymp_mean_per_run)

# ----------- 主分析流程 -----------

# 将CSV文件模式与CSV_DIR拼接
gamma90_runs = load_group(os.path.join(CSV_DIR, gamma90_pattern))
gamma95_runs = load_group(os.path.join(CSV_DIR, gamma95_pattern))
gamma99_runs = load_group(os.path.join(CSV_DIR, gamma99_pattern))

x_g90, y_g90 = align_curves(gamma90_runs)
x_g95, y_g95 = align_curves(gamma95_runs)
x_g99, y_g99 = align_curves(gamma99_runs)

# 应用一个更专业的绘图风格
sns.set_theme(style="darkgrid")

plt.figure(figsize=(10, 6))

# 绘制曲线，让绘图函数内部自动处理颜色同步
if x_g90.size > 0: plot_learning_curve(x_g90, y_g90, label="Gamma = 0.90")
if x_g95.size > 0: plot_learning_curve(x_g95, y_g95, label="Gamma = 0.95")
if x_g99.size > 0: plot_learning_curve(x_g99, y_g99, label="Gamma = 0.99")

plt.title("Learning Curve Comparison for Different Gamma Values", fontsize=16)
plt.xlabel("Training Timesteps", fontsize=14)
plt.ylabel("Success Rate", fontsize=14)
plt.legend(fontsize=12)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
plt.show()

