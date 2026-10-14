import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")

df = pd.read_excel('test-data-spatial.xlsx')

goal_xy = df.iloc[:, 0].astype(float)
mean_std = df.iloc[:, 1].str.split('±', expand=True)
mean = mean_std[0].astype(float)
std = mean_std[1].astype(float)

plt.figure(figsize=(10, 6))
plt.errorbar(goal_xy, mean, yerr=std, fmt='o-', capsize=4, color='tab:blue', ecolor='gray', elinewidth=2, markeredgewidth=2)
plt.xlabel('Goal XY Range', fontsize=14)
plt.ylabel('Success Rate (%)', fontsize=14)
plt.title('Spatial Generalization Success Rate (Mean±Std)', fontsize=16)
plt.ylim(0, 110)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('spatial_generalization_point.png', dpi=200)
plt.show()
