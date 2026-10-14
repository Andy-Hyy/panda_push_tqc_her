# train_gamma.py

import os
import sys
import time
import argparse
import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.her import HerReplayBuffer

import panda_mujoco_gym 

def train(seed, gamma):
    """
    一个包含完整训练逻辑的函数，接收种子和gamma值作为输入。
    其他超参数被固定为“冠军配置”（HIT配置）。
    """
    # --- 固定超参数 ---
    batch_size = 512
    n_sampled_goal = 4
    env_id = "FrankaPushSparse-v0"

    print(f"--- 开始 Gamma 敏感性分析训练 ---")
    print(f"固定配置: Batch Size={batch_size}, HER Goals={n_sampled_goal}")
    print(f"可变参数: Seed={seed}, Gamma={gamma}")

    # --- 1. 创建环境 ---
    env = gym.make(env_id)

    # --- 2. 设置独立的日志和模型路径 ---
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # 日志名现在包含gamma值，方便区分
    config_name = f"gamma{gamma}"
    log_dir = f"logs/{env_id}_TQC_gamma_test/{config_name}_seed{seed}_{run_timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    
    final_model_path = os.path.join(log_dir, "final_model.zip")

    # --- 3. 创建回调 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(log_dir, "checkpoints/"),
        name_prefix=f"tqc_frankapush_gamma{gamma}",
    )

    # --- 4. 创建TQC模型 ---
    model = TQC(
        policy="MultiInputPolicy",
        env=env,
        seed=seed,
        learning_rate=1e-3,
        buffer_size=1000000,
        batch_size=batch_size, # 使用固定的冠军配置参数
        tau=0.05,
        gamma=gamma, # <<< 核心修正：使用从命令行传入的gamma值
        top_quantiles_to_drop_per_net=2,
        ent_coef="auto",
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": n_sampled_goal, # 使用固定的冠军配置参数
            "goal_selection_strategy": "future",
        },
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]),
            n_quantiles=25
        ),
        tensorboard_log=log_dir,
        verbose=1
    )

    # --- 5. 开始学习 ---
    print(f"日志将保存在: {log_dir}")
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save(final_model_path)
    print(f"\n训练完成！模型已保存至: {final_model_path}")


if __name__ == "__main__":
    # --- 6. 使用 argparse 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="为FrankaPush任务训练TQC模型(Gamma敏感性测试)")
    # 只保留 seed 和 gamma 作为可变参数
    parser.add_argument("--seed", type=int, default=0, help="训练使用的随机种子")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子 Gamma")
    
    args = parser.parse_args()

    # 调用主训练函数
    train(seed=args.seed, gamma=args.gamma)