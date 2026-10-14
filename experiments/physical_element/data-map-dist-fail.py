import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid", font_scale=1.25)

df = pd.read_csv('multi_param_summary.csv')
print(df.columns)  # 打印所有列名

# 映射实际列名（这里需要你自己查一下实际文件内容）
rename_map = {
    'object_mass': 'mass',
    'object_friction': 'friction',
    'joint_damping_scale': 'damping_scale',
    'delay_steps': 'delay_steps',  # 可能已经对了
    # 根据实际情况补充
}
df = df.rename(columns=rename_map)

df['outcome'] = df['success'].map({1: 'Success', 0: 'Fail'})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(['mass', 'friction', 'damping_scale', 'delay_steps']):
    ax = axes[idx // 2, idx % 2]
    sns.histplot(data=df, x=col, hue='outcome', multiple='stack', palette='Set1', ax=ax, bins=12, alpha=0.7)
    ax.set_title(f'{col.capitalize()} Distribution by Outcome')
plt.suptitle('Distribution of Physical Parameters by Success/Fail (Multi-Parameter Disturbance)', fontsize=16)
plt.tight_layout()
plt.savefig('multi_param_dist_success_fail.png')
plt.show()
