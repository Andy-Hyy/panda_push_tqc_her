import os
os.chdir('/Users/hejin/Documents/panda_multi_element_test/')  # 保证路径正确
import numpy as np
import random
import torch
import pandas as pd
from sb3_contrib import TQC
from domain_rand import DomainRandomizationWrapper


MODEL_PATH = "logs/std_model.zip"  # 改为你的模型路径
N_EPISODES = 200
SEEDS = [123]

# 设定物理参数联合扰动范围
MASS_RANGE = (0.05, 3.0)
FRICTION_RANGE = (0.01, 3.0)
DAMPING_RANGE = (0.1, 5.0)
DELAY_RANGE = (0, 10)  # 10步以内的控制延迟
GOAL_RANGE = {'low': [-0.15, -0.15, 0.0], 'high': [0.15, 0.15, 0.0]}

for seed in SEEDS:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env = DomainRandomizationWrapper(
        reward_type='sparse',
        goal_range=GOAL_RANGE,
        mass_range=MASS_RANGE,
        friction_range=FRICTION_RANGE,
        damping_range=DAMPING_RANGE,
        delay_range=DELAY_RANGE,
        max_episode_steps=100
    )
    obs, info = env.reset(seed=seed)
    model = TQC.load(MODEL_PATH, env=env)
    logs = []
    success_cnt = 0
    for i in range(N_EPISODES):
        obs, info = env.reset()
        params = env.last_params  # 获取本轮物理扰动参数
        done = False
        steps = 0
        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            done = done or trunc
            steps += 1
        is_success = int(reward == 0)
        success_cnt += is_success
        logs.append({
            "episode": i+1,
            "mass": params["mass"],
            "friction": params["friction"],
            "damping": params["damping"],
            "delay_steps": params["delay_steps"],
            "goal_noise_x": params["goal_noise"][0],
            "goal_noise_y": params["goal_noise"][1],
            "goal_noise_z": params["goal_noise"][2],
            "success": is_success
        })
    sr = success_cnt / N_EPISODES
    print(f"Seed {seed}: Success rate = {sr*100:.1f}%")
    df = pd.DataFrame(logs)
    df.to_csv(f"multi_param_disturbance_wrapper_seed{seed}.csv", index=False)

print("\n全部seed的联合扰动鲁棒性测试完成，结果已保存为csv！")
