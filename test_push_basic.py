import gymnasium as gym
from sb3_contrib import TQC
import numpy as np
import argparse

import panda_mujoco_gym

def evaluate_model(model_path: str, num_episodes: int, seed: int):
    if not model_path:
        print("错误：必须提供模型文件路径。请使用 --model-path 参数。")
        return

    print("--- 开始模型评估 ---")
    print(f"模型路径: {model_path}")
    print(f"评估回合数: {num_episodes}")
    print(f"使用随机种子: {seed}")

    # 1. 创建环境，并设置seed
    try:
        env = gym.make("FrankaPushSparse-v0",render_mode="human")
        env.reset(seed=seed)  # 保证环境随机性复现
        np.random.seed(seed)
        print("✅ 标准环境 'FrankaPushSparse-v0' 创建成功。")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        print("请确保您已经通过 'pip install -e .' 正确安装了您的 panda_mujoco_gym 包。")
        return

    # 2. 加载模型
    try:
        model = TQC.load(model_path, env=env)
        print("✅ 模型加载成功。")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请检查模型路径是否正确，以及模型算法（如TQC）是否与加载时一致。")
        env.close()
        return

    # 3. 评估循环
    successes = 0
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)  # 保证每一局都不同，但可控
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        if info.get('is_success'):
            successes += 1
        print(f"  > 回合 {i + 1}/{num_episodes} 完成... 状态: {'成功' if info.get('is_success') else '失败'}")

    success_rate = (successes / num_episodes) * 100
    print("\n--- 评估完成 ---")
    print(f"总回合数: {num_episodes}")
    print(f"成功次数: {successes}")
    print(f"成功率: {success_rate:.2f}%")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估一个已训练的Panda Push模型。")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="指向已保存的.zip模型文件的路径（例如：'logs/best_model.zip'）。"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="用于评估的总回合数，默认为100。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="用于环境和随机数生成器的随机种子，默认为0。"
    )
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path, num_episodes=args.episodes, seed=args.seed)
