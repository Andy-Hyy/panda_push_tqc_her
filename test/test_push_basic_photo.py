import gymnasium as gym
from sb3_contrib import TQC
import numpy as np
import argparse
import os
from PIL import Image

import panda_mujoco_gym

def evaluate_model(model_path: str, num_episodes: int, seed: int):
    if not model_path:
        print("错误：必须提供模型文件路径。请使用 --model-path 参数。")
        return

    print("--- 开始模型评估（带高分辨率帧保存功能） ---")
    print(f"模型路径: {model_path}")
    print(f"评估回合数: {num_episodes}")
    print(f"使用随机种子: {seed}")

    # --- (关键修改) 在创建环境时指定宽度和高度 ---
    try:
        env = gym.make(
            "FrankaPushSparse-v0", 
            render_mode="rgb_array",
            width=1280, 
            height=720
        )
        env.reset(seed=seed)
        np.random.seed(seed)
        print("✅ 标准环境 'FrankaPushSparse-v0' (1280x720 rgb_array模式) 创建成功。")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return

    # 加载模型
    try:
        model = TQC.load(model_path, env=env)
        print("✅ 模型加载成功。")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        env.close()
        return

    # 创建用于存放图片的文件夹
    output_dir = "evaluation_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"截图将保存在 '{output_dir}/' 目录下。")

    # 评估循环
    successes = 0
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        
        # 在回合结束后渲染并保存画面
        frame = env.render()
        img = Image.fromarray(frame)
        
        outcome = 'success' if info.get('is_success') else 'fail'
        img.save(os.path.join(output_dir, f"episode_{i+1:03d}_{outcome}.png"))

        if info.get('is_success'):
            successes += 1
        print(f"  > 回合 {i + 1}/{num_episodes} 完成... 状态: {outcome.capitalize()} (截图已保存)")

    success_rate = (successes / num_episodes) * 100
    print("\n--- 评估完成 ---")
    print(f"总回合数: {num_episodes}")
    print(f"成功次数: {successes}")
    print(f"成功率: {success_rate:.2f}%")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估一个已训练的Panda Push模型并保存最终帧。")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="指向已保存的.zip模型文件的路径（例如：'logs/best_model.zip'）。"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="用于评估的总回合数，默认为10。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="用于环境和随机数生成器的随机种子，默认为0。"
    )
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path, num_episodes=args.episodes, seed=args.seed)