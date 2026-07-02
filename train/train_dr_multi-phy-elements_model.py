import os
import random
import numpy as np
import torch
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# 导入准备好的Wrapper
from utils.domain_rand import DomainRandomizationWrapper

# --- 训练配置 ---
SEEDS_TO_TRAIN = [0, 42, 123] 
N_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 50_000 
LEARNING_STARTS = 1000

# --- 领域随机化（DR）的参数范围 ---
MASS_RANGE = (0.05, 3.0)
FRICTION_RANGE = (0.05, 3.0)
DAMPING_SCALE_RANGE = (0.001, 0.3)
ACTUATOR_GAIN_SCALE_RANGE = (0.0, 10.0)
DELAY_RANGE = (1, 10)
GOAL_RANGE = {'low': [-0.15, -0.15, 0.0], 'high': [0.15, 0.15, 0.0]}

def main():
    """主函数，循环遍历所有种子进行独立的并行训练"""

    for seed in SEEDS_TO_TRAIN:
        print(f"\n{'='*25} 开始训练 Seed: {seed} {'='*25}\n")

        log_dir = f"./logs/dr_model_seed{seed}"
        os.makedirs(log_dir, exist_ok=True)

        env_kwargs = dict(
            reward_type='sparse',
            goal_range=GOAL_RANGE,
            mass_range=MASS_RANGE,
            friction_range=FRICTION_RANGE,
            damping_scale_range=DAMPING_SCALE_RANGE,
            gain_scale_range=ACTUATOR_GAIN_SCALE_RANGE,
            delay_range=DELAY_RANGE,
            max_episode_steps=100
        )

        print(f"--- 正在创建 {N_ENVS} 个并行的训练环境 (SubprocVecEnv)... ---")
        env = make_vec_env(
            DomainRandomizationWrapper,
            n_envs=N_ENVS,
            seed=seed,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs
        )

        model_params = {
            "policy": "MultiInputPolicy",
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs": dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            "verbose": 1,
            "batch_size": 512,
            "learning_starts": LEARNING_STARTS,
            "tensorboard_log": f"{log_dir}/tensorboard/"
        }

        print("--- 正在初始化TQC模型 ---")
        model = TQC(env=env, **model_params)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(SAVE_FREQ // N_ENVS, 1), 
            save_path=log_dir,
            name_prefix="rl_model"
        )

        print(f"--- 开始训练，总步数: {TOTAL_TIMESTEPS} ---")
        # 关键修正：添加 progress_bar=True 来显示进度条
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback, 
            progress_bar=True
        )
        
        final_model_path = os.path.join(log_dir, "final_model.zip")
        model.save(final_model_path)
        
        env.close()
        
        print(f"\n🎉 Seed {seed} 训练完成！最终模型已保存至: {final_model_path}\n")

if __name__ == "__main__":
    main()