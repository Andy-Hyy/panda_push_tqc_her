import os
import sys  # 修复 #1: 添加此行以使用 sys.stderr
import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
import panda_mujoco_gym  # 修复 #2: 添加此行以注册自定义环境

# 此脚本被主脚本调用，在隔离的进程中运行测试

# 硬编码配置，因为此脚本不应独立更改
MODEL_PATH = "logs/FrankaPushSparse_v0_TQC/batch512_her4_seed0_2025-06-17_13-50-58/std_model.zip"
SEEDS = [0, 42, 123]
NUM_EPISODES_PER_SEED = 200
ENV_ID = "FrankaPushSparse-v0"

def run_evaluation_episodes(model, env, num_episodes, seed):
    """
    在已创建的环境中运行评估。
    """
    successful_episodes = 0
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        terminated, truncated = False, False
        while not (bool(terminated) or bool(truncated)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get("is_success"):
            successful_episodes += 1
            
    return (successful_episodes / num_episodes) * 100

def main():
    """
    工作者脚本的主函数。
    加载模型，创建环境，为所有种子运行评估，然后将结果打印到 stdout。
    """
    # 加载模型需要一个临时环境
    try:
        temp_env_for_loading = gym.make(ENV_ID)
        model = TQC.load(MODEL_PATH, env=temp_env_for_loading)
        temp_env_for_loading.close()
    except Exception as e:
        print(f"Worker Error: Failed to load model. {e}", file=sys.stderr)
        return

    # 创建用于实际评估的环境
    try:
        eval_env = gym.make(ENV_ID)
        
        all_rates = []
        for seed in SEEDS:
            rate = run_evaluation_episodes(model, eval_env, NUM_EPISODES_PER_SEED, seed)
            all_rates.append(rate)
        
        eval_env.close()

        # 将结果打印到 stdout，每行一个，供主脚本捕获
        for rate in all_rates:
            print(f"{rate:.2f}")

    except Exception as e:
        # 将错误信息打印到 stderr，以便主脚本可以捕获它
        print(f"Worker Error: Evaluation failed. {e}", file=sys.stderr)
        return

if __name__ == "__main__":
    main()