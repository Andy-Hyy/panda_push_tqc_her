import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

import numpy as np
import random
import torch
import pandas as pd
import argparse
from sb3_contrib import TQC
import panda_mujoco_gym
from domain_rand import DomainRandomizationWrapper

# --- 常量定义 ---
SEEDS = [0, 42, 123]
N_EPISODES_PER_SEED = 200
MAX_EPISODE_STEPS = 100

# --- 与单因素测试范围完全对齐 ---
MASS_RANGE = (0.05, 3.0)
FRICTION_RANGE = (0.05, 3.0)
DAMPING_SCALE_RANGE = (0.001, 0.3)
ACTUATOR_GAIN_SCALE_RANGE = (0.0, 10.0)
# ---------------------------------------------
# 以下是多因素测试独有的扰动
DELAY_RANGE = (1, 10)
GOAL_RANGE = {'low': [-0.15, -0.15, 0.0], 'high': [0.15, 0.15, 0.0]}

def run_multi_param_evaluation(model_path, output_csv):
    """对单个模型运行多参数联合扰动测试"""
    all_logs = []
    
    print(f"\n[Worker] 开始对模型 {model_path} 进行多参数联合扰动测试...")

    print("[Worker] 正在创建唯一的、持久化的评测环境...")
    env = None
    try:
        # 使用修正后的参数范围创建Wrapper
        env = DomainRandomizationWrapper(
            reward_type='sparse',
            goal_range=GOAL_RANGE,
            mass_range=MASS_RANGE,
            friction_range=FRICTION_RANGE,
            damping_scale_range=DAMPING_SCALE_RANGE,
            gain_scale_range=ACTUATOR_GAIN_SCALE_RANGE,
            delay_range=DELAY_RANGE,
            max_episode_steps=MAX_EPISODE_STEPS
        )
        
        print("[Worker] 正在加载模型...")
        model = TQC.load(model_path, env=env)
        
        for seed in SEEDS:
            print(f"[Worker] 正在评测 Seed = {seed}...")
            
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            
            for i in range(N_EPISODES_PER_SEED):
                obs, info = env.reset()
                params = env.last_params
                
                terminated, truncated = False, False
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                
                is_success = 1 if info.get("is_success") else 0
                log_entry = {
                    "seed": seed, "episode": i + 1, "mass": params.get("mass"),
                    "friction": params.get("friction"), "damping_scale": params.get("damping_scale"),
                    "gain_scale": params.get("gain_scale"),"delay_steps": params.get("delay_steps"),
                    "goal_noise_x": params.get("goal_noise")[0],
                    "goal_noise_y": params.get("goal_noise")[1],
                    "goal_noise_z": params.get("goal_noise")[2],
                    "success": is_success
                }
                all_logs.append(log_entry)
            
        print(f"[Worker] 模型 {os.path.basename(model_path)} 的所有seed评测完成。")

    finally:
        if env:
            print("[Worker] 正在关闭环境...")
            env.close()

    # 将所有详细日志一次性写入CSV文件
    df = pd.DataFrame(all_logs)
    # 确保列的顺序方便查看
    columns_order = [
        "seed", "episode", "success", "mass", "friction", 
        "damping_scale", "gain_scale", "delay_steps",
        "goal_noise_x", "goal_noise_y", "goal_noise_z"
    ]
    df = df[columns_order]
    df.to_csv(output_csv, index=False)
    print(f"[Worker] 评测结果已保存到 {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Element Generalization Worker")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()
    run_multi_param_evaluation(args.model_path, args.output_csv)