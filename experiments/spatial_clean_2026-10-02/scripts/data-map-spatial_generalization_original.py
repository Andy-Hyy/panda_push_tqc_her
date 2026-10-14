import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Set style consistent with RL learning curve ---
sns.set(style="darkgrid", font_scale=1.25)

# Read csv
try:
    df = pd.read_csv('spatial_generalization.csv', index_col=0, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('spatial_generalization.csv', index_col=0, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv('spatial_generalization.csv', index_col=0, encoding='ISO-8859-1')

# Convert to fraction if contains '%'
if '%' in str(df.values[0][0]):
    df = df.applymap(lambda x: float(str(x).replace('%',''))/100)

goal_ranges = [float(col) for col in df.columns]
seeds = df.index.astype(str)

plt.figure(figsize=(8, 5))
palette = sns.color_palette("Set2", n_colors=len(seeds))

# Plot each seed
for i, seed in enumerate(seeds):
    plt.plot(goal_ranges, df.loc[seed]*100, marker='o', label=f'Seed {seed}', color=palette[i])

mean_curve = df.astype(float).mean(axis=0)*100
std_curve = df.astype(float).std(axis=0)*100
plt.plot(goal_ranges, mean_curve, color='k', linestyle='--', label='Mean', linewidth=2)
plt.fill_between(goal_ranges, mean_curve-std_curve, mean_curve+std_curve,
                 color='gray', alpha=0.17, label='Std')

plt.xlabel('Goal XY Range', fontsize=14)
plt.ylabel('Success Rate (%)', fontsize=14)
plt.title('Spatial Generalization Success Rate', fontsize=16)
plt.ylim(0, 105)
plt.legend()
plt.tight_layout()
plt.savefig('spatial_generalization_en_style.png', dpi=300)
plt.show()
