import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", font_scale=1.25)

file_info = {
    "Object Mass": "eval_phy_object_mass.csv",
    "Object Friction": "eval_phy_object_friction.csv",
    "Joint Damping": "eval_phy_joint_damping_scale.csv",
    "Actuator Gain": "eval_phy_actuator_gain_scale.csv"
}

xlabels = {
    "Object Mass": "Object Mass",
    "Object Friction": "Object Friction",
    "Joint Damping": "Joint Damping Scale",
    "Actuator Gain": "Actuator Gain Scale"
}

for pname, fname in file_info.items():
    df = pd.read_csv(fname)
    df.columns = df.columns.str.strip()
    print(f"{fname} columns: {df.columns.tolist()}")

    # 自动找参数值列
    if 'value' in df.columns:
        x_col = 'value'
    elif 'param_value' in df.columns:
        x_col = 'param_value'
    else:
        x_col = df.columns[0]  # 最保险兜底

    # 自动找seed列
    if 'seed' in df.columns:
        seed_col = 'seed'
    elif 'model seed' in df.columns:
        seed_col = 'model seed'
    else:
        seed_col = None

    plt.figure(figsize=(8, 5))
    if seed_col:
        for seed in sorted(df[seed_col].unique()):
            tmp = df[df[seed_col] == seed]
            plt.plot(tmp[x_col], tmp['success_rate']*100, marker='o', label=f'Seed {seed}', alpha=0.7)
        group = df.groupby(x_col)['success_rate']
        mean = group.mean()*100
        std = group.std()*100
        plt.errorbar(mean.index, mean, yerr=std, fmt='o--', color='black', capsize=4, label='Mean ± Std')
    else:
        # 只有均值，不分seed，自动用success_rate
        plt.plot(df[x_col], df['success_rate']*100, 'o--', color='black', label='Success Rate')

    plt.xlabel(xlabels[pname], fontsize=13)
    plt.ylabel('Success Rate (%)', fontsize=13)
    plt.title(f'Success Rate under {pname} Disturbance', fontsize=15)
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()
    outname = f'{pname.replace(" ", "_").lower()}_disturbance.png'
    plt.savefig(outname, dpi=200)
    plt.close()
print("All figures saved!")
