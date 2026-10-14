import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 定义不同实验组的日志目录
CSV_DIR_GAMMA_TEST = "logs/FrankaPushSparse_v0_TQC_gamma_test/Tables"
CSV_DIR_BASELINE = "logs/FrankaPushSparse_v0_TQC/tables"

# 为不同的gamma值定义正确的文件名匹配模式
gamma_090_pattern = "gamma0.9_seed*_*_TQC_1.csv"
gamma_095_pattern = "FrankaPushSparse_v0_TQC_batch512_her4_seed*_*_TQC_1.csv"
gamma_099_pattern = "gamma0.99_seed*_*_TQC_1.csv"

# 定义输出文件名
TABLE_OUTPUT_PATH = "gamma_performance_summary.csv"
FIGURE_OUTPUT_PATH = "LearningCurve_Gamma_Comparison.png"


def load_group(csv_pattern, base_dir):
    """Load all csv files matching pattern, return aligned arrays of steps & success_rate"""
    full_pattern = os.path.join(base_dir, csv_pattern)
    csv_files = glob.glob(full_pattern)
    if not csv_files:
        print(f"警告：找不到任何CSV文件，匹配模式为: {full_pattern}")
        return []
    print(f"为模式 '{csv_pattern}' 加载了 {len(csv_files)} 个运行: {csv_files}")
    runs = []
    for file in csv_files:
        df = pd.read_csv(file)
        step_col = [c for c in df.columns if "step" in c.lower()][0]
        sr_col = [c for c in df.columns if "success" in c.lower()][0]
        runs.append((df[step_col].to_numpy(), df[sr_col].to_numpy()))
    return runs

def align_curves(runs, num_points=500):
    """Interpolate all runs to a common x axis (for averaging curves)"""
    if not runs:
        return None, None
    max_step = min(r[0].max() for r in runs)
    min_step = max(r[0].min() for r in runs)
    common_x = np.linspace(min_step, max_step, num_points)
    aligned_y = []
    for x, y in runs:
        aligned = np.interp(common_x, x, y)
        aligned_y.append(aligned)
    return common_x, np.vstack(aligned_y)

def plot_learning_curve(x, y_mat, label, color):
    """Plot mean learning curve with shaded standard deviation."""
    mean_y = np.mean(y_mat, axis=0)
    std_y = np.std(y_mat, axis=0)
    plt.plot(x, mean_y, label=label, linewidth=2.5, color=color)
    plt.fill_between(x, np.clip(mean_y - std_y, 0, 1), np.clip(mean_y + std_y, 0, 1), 
                     color=color, alpha=0.18)

def calc_sample_efficiency(x, mean_y, threshold=0.8):
    """Calculate timesteps to reach a certain success rate threshold."""
    if x is None or mean_y is None: return np.nan
    idx = np.where(mean_y >= threshold)[0]
    return int(x[idx[0]]) if len(idx) > 0 else np.nan

def calc_asymptotic(y_mat, portion=0.2):
    """Calculate asymptotic performance on the final portion of the curve."""
    if y_mat is None: return np.nan, np.nan
    n = y_mat.shape[1]
    last_n = int(n * portion)
    asymp_mean = np.mean(y_mat[:, -last_n:], axis=1)
    return np.mean(asymp_mean), np.std(asymp_mean)

print("--- 开始分析Gamma值对性能的影响 ---")

runs_090 = load_group(gamma_090_pattern, base_dir=CSV_DIR_GAMMA_TEST)
runs_095 = load_group(gamma_095_pattern, base_dir=CSV_DIR_BASELINE)
runs_099 = load_group(gamma_099_pattern, base_dir=CSV_DIR_GAMMA_TEST)

x_090, y_090 = align_curves(runs_090)
x_095, y_095 = align_curves(runs_095)
x_099, y_099 = align_curves(runs_099)

plt.figure(figsize=(10, 7))
if x_090 is not None:
    plot_learning_curve(x_090, y_090, label="Gamma = 0.90", color="#E69F00")
if x_095 is not None:
    plot_learning_curve(x_095, y_095, label="Gamma = 0.95 (Baseline)", color="#56B4E9")
if x_099 is not None:
    plot_learning_curve(x_099, y_099, label="Gamma = 0.99", color="#009E73")

plt.xlabel("Timesteps", fontsize=14)
plt.ylabel("Success Rate", fontsize=14)
plt.title("Impact of Gamma on Learning Performance (FrankaPush-TQC+HER)", fontsize=16)
plt.legend(fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(FIGURE_OUTPUT_PATH, dpi=200)
print(f"\n学习曲线图已保存至: {FIGURE_OUTPUT_PATH}")
plt.show()

mean_090 = np.mean(y_090, axis=0) if y_090 is not None else None
mean_095 = np.mean(y_095, axis=0) if y_095 is not None else None
mean_099 = np.mean(y_099, axis=0) if y_099 is not None else None

eff80_090 = calc_sample_efficiency(x_090, mean_090, threshold=0.8)
eff80_095 = calc_sample_efficiency(x_095, mean_095, threshold=0.8)
eff80_099 = calc_sample_efficiency(x_099, mean_099, threshold=0.8)

eff95_090 = calc_sample_efficiency(x_090, mean_090, threshold=0.95)
eff95_095 = calc_sample_efficiency(x_095, mean_095, threshold=0.95)
eff95_099 = calc_sample_efficiency(x_099, mean_099, threshold=0.95)

asymp_mean_090, asymp_std_090 = calc_asymptotic(y_090)
asymp_mean_095, asymp_std_095 = calc_asymptotic(y_095)
asymp_mean_099, asymp_std_099 = calc_asymptotic(y_099)

summary = pd.DataFrame({
    "Config (Gamma)": ["0.90", "0.95 (Baseline)", "0.99"],
    "SampleEff_80%_steps": [eff80_090, eff80_095, eff80_099],
    "SampleEff_95%_steps": [eff95_090, eff95_095, eff95_099],
    "Asymptotic_Mean": [asymp_mean_090, asymp_mean_095, asymp_mean_099],
    "Asymptotic_Std": [asymp_std_090, asymp_std_095, asymp_std_099]
})

print("\n===== Gamma Performance Comparison Summary =====\n")
print(summary.to_string(index=False))
summary.to_csv(TABLE_OUTPUT_PATH, index=False)
print(f"\n性能总结表格已保存至: {TABLE_OUTPUT_PATH}")