import os
import sys
import time
import argparse
import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import panda_mujoco_gym

N_ENVS = 4
LEARNING_STARTS = 1000 # 1000步才能开始学习，否则并行环境没有都准备好

def train(seed, batch_size, n_sampled_goal):
    print(f"--- 开始新一轮并行训练 ---")
    print(f"随机种子 (Seed): {seed}, 批量大小 (Batch Size): {batch_size}, HER采样数: {n_sampled_goal}")
    print(f"并行环境数 (N_ENVS): {N_ENVS}, 学习开始步数 (Learning Starts): {LEARNING_STARTS}")

    env = make_vec_env(
        "FrankaPushSparse-v0",
        n_envs=N_ENVS,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(reward_type='sparse')
    )

    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    config_name = f"batch{batch_size}_her{n_sampled_goal}"
    log_dir = f"logs/FrankaPushSparse_v0_TQC_Parallel/{config_name}_seed{seed}_{run_timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    
    final_model_path = os.path.join(log_dir, "final_model.zip")

    save_freq_per_env = max(10000 // N_ENVS, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=os.path.join(log_dir, "checkpoints/"),
        name_prefix="tqc_frankapush_parallel",
    )

    model = TQC(
        policy="MultiInputPolicy",
        env=env,
        seed=seed,
        learning_rate=1e-3,
        buffer_size=1000000,
        batch_size=batch_size,
        tau=0.05,
        gamma=0.95,
        top_quantiles_to_drop_per_net=2,
        ent_coef="auto",
        learning_starts=LEARNING_STARTS, 
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": n_sampled_goal,
            "goal_selection_strategy": "future",
        },
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]),
            n_quantiles=25
        ),
        tensorboard_log=log_dir,
        verbose=1
    )

    print(f"日志将保存在: {log_dir}")
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save(final_model_path)
    print(f"\n训练完成！模型已保存至: {final_model_path}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为FrankaPush任务并行训练TQC模型")
    parser.add_argument("--seed", type=int, default=0, help="训练使用的随机种子")
    parser.add_argument("--batch_size", type=int, default=512, help="批量大小")
    parser.add_argument("--n_sampled_goal", type=int, default=4, help="HER采样数")

    args = parser.parse_args()
    train(seed=args.seed, batch_size=args.batch_size, n_sampled_goal=args.n_sampled_goal)