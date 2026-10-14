# multi_param_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. 读取三组seed的csv，添加seed列，合并
df0 = pd.read_csv('eval_multi_param_seed0.csv')
df42 = pd.read_csv('eval_multi_param_seed42.csv')
df123 = pd.read_csv('eval_multi_param_seed123.csv')
df0['seed'] = 0
df42['seed'] = 42
df123['seed'] = 123
df_all = pd.concat([df0, df42, df123], ignore_index=True)

# 2. 增加 outcome 列（用于分组绘图）
df_all['outcome'] = df_all['success'].map({1: 'Success', 0: 'Fail'})

# 3. 汇总统计（可打印/保存）
summary = df_all.groupby('seed')['success'].agg(['count', 'sum'])
summary['fail'] = summary['count'] - summary['sum']
summary['success_rate'] = summary['sum'] / summary['count']
print('按随机种子的成功率统计：')
print(summary)
print('\n三组seed平均成功率: %.2f%%\n' % (df_all['success'].mean() * 100))

summary.to_csv('multi_param_generalization_summary.csv', index=True)

# 4. 绘制参数分布直方图（成功/失败对比）
sns.set(style="darkgrid")
param_list = ['mass', 'friction', 'damping_scale', 'gain_scale', 'delay_steps']
for param in param_list:
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df_all, x=param, hue='outcome', multiple='stack', palette='Set1', bins=14, alpha=0.7)
    plt.title(f'Distribution of {param.replace("_", " ").capitalize()} by Success/Fail', fontsize=13)
    plt.xlabel(param.replace('_', ' ').capitalize())
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'dist_{param}_success_fail_en.png')
    plt.close()

# 5. 联合参数散点图举例（可扩展为其它组合）
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df_all, x='mass', y='friction', hue='outcome', palette='Set1', alpha=0.65)
plt.title('Success vs. Fail in Mass-Friction Space')
plt.xlabel('Mass')
plt.ylabel('Friction')
plt.tight_layout()
plt.savefig('scatter_mass_friction_en.png')
plt.close()

# 6. 可选：失败案例参数分布单独输出
fail_df = df_all[df_all['success'] == 0]
fail_df[param_list + ['seed', 'episode']].to_csv('multi_param_failed_cases.csv', index=False)
print('失败案例已保存为 multi_param_failed_cases.csv')

print('\n所有图像、表格已生成。可用于论文、PPT分析。')
