import pandas as pd

file_info = {
    "object_mass": "eval_phy_object_mass.csv",
    "object_friction": "eval_phy_object_friction.csv",
    "joint_damping": "eval_phy_joint_damping_scale.csv",
    "actuator_gain": "eval_phy_actuator_gain_scale.csv"
}

for key, fname in file_info.items():
    df = pd.read_csv(fname)
    # 自动找参数列
    param_col = [col for col in df.columns if col not in ["seed", "success", "fail", "success_rate", "param_name", "model seed"]][-1]
    summary = df.groupby(param_col)["success_rate"].agg(["mean", "std"]).reset_index()
    summary["Mean Success Rate (%)"] = (summary["mean"] * 100).round(1)
    summary["Std (%)"] = (summary["std"] * 100).round(1)
    # 拼接时用"+/-"
    summary["Mean +/- Std (%)"] = summary["Mean Success Rate (%)"].astype(str) + " +/- " + summary["Std (%)"].astype(str)
    summary = summary.rename(columns={param_col: key.replace("_", " ").title()})
    summary = summary[[key.replace("_", " ").title(), "Mean Success Rate (%)", "Std (%)", "Mean +/- Std (%)"]]
    outname = f"{key}_summary_clean.csv"
    summary.to_csv(outname, index=False)
    print(f"Saved clean table: {outname}")
