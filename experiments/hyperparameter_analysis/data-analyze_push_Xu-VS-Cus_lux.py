import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_dir = "./logs/FrankaPushSparse_v0_TQC/tables/"
files_512 = sorted(glob.glob(os.path.join(csv_dir, "*batch512_her4_seed*_TQC_1.csv")))
files_256 = sorted(glob.glob(os.path.join(csv_dir, "*batch256_her8_seed*_TQC_1.csv")))

groups = {
    "Xu config (batch=512, k=4)": files_512,
    "Customized (batch=256, k=8)": files_256
}

data = {}
for group, filelist in groups.items():
    curves = []
    for f in filelist:
        df = pd.read_csv(f)
        step_col = [c for c in df.columns if 'step' in c.lower()][0]
        sr_col = [c for c in df.columns if 'success' in c.lower() or 'value' in c.lower()][0]
        curves.append((df[step_col].values, df[sr_col].values))
    # 对齐曲线步数
    min_step = max(c[0][0] for c in curves)
    max_step = min(c[0][-1] for c in curves)
    num_points = 2000
    common_x = np.linspace(min_step, max_step, num_points)
    ys = [np.interp(common_x, c[0], c[1]) for c in curves]
    data[group] = (common_x, np.vstack(ys))

sns.set(style="darkgrid", font_scale=1.25)
plt.figure(figsize=(8, 5))

for group, (x, y) in data.items():
    mean = np.mean(y, axis=0) * 100
    std = np.std(y, axis=0) * 100
    line = plt.plot(x, mean, label=group, linewidth=2)[0]
    plt.fill_between(x, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100),
                     color=line.get_color(), alpha=0.18)

plt.xlabel("Timesteps", fontsize=14)
plt.ylabel("Success Rate (%)", fontsize=14)
plt.title("Learning Curve Comparison on FrankaPushSparse-v0-TQC", fontsize=15)
plt.legend(fontsize=13)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig("learning_curve_Xu_vs_Customized.png", dpi=300)
plt.show()

def get_performance_summary(x, y, thres=[0.8, 0.9], portion=0.2):
    mean_curve = y.mean(axis=0)
    results = []
    for t in thres:
        idx = np.where(mean_curve >= t)[0]
        results.append(int(x[idx[0]]) if len(idx) > 0 else np.nan)
    n = y.shape[1]
    last = int(n * portion)
    final_mean = np.mean(y[:, -last:], axis=1)
    return results + [final_mean.mean() * 100, final_mean.std() * 100]

summary_rows = []
for group, (x, y) in data.items():
    summary_rows.append(get_performance_summary(x, y))

summary = pd.DataFrame(
    summary_rows,
    columns=["80%_step", "90%_step", "FinalMean(%)", "FinalStd(%)"],
    index=list(groups.keys())
)
summary.to_csv("performance_summary_Xu_vs_Customized.csv", float_format="%.2f")
print("\n==== Performance Summary (for paper) ====")
print(summary.to_string(float_format="%.2f"))
