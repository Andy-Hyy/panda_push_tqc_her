import os
import random
import numpy as np
import torch
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils.domain_rand import DomainRandomizationWrapper

# --- 训练配置 ---
SEEDS_TO_TRAIN = [0, 42, 123] 
N_ENVS = 6
TOTAL_TIMESTEPS = 2_000_000 # 建议使用更长的训练时间以适应更长的课程
SAVE_FREQ = 50_000 
LEARNING_STARTS = 50000

# --- 课程学习：定义更平滑的、5个难度等级的随机化范围 ---

CURRICULUM_1_EASY = {
    "mass_range": (0.9, 1.1), "friction_range": (0.9, 1.1),
    "damping_scale_range": (0.8, 1.2), "gain_scale_range": (0.8, 1.2),
    "delay_range": (1, 2), "goal_range": {'low': [-0.05, -0.05, 0.0], 'high': [0.05, 0.05, 0.0]},
}

CURRICULUM_2_EASY_PLUS = {
    "mass_range": (0.7, 1.3), "friction_range": (0.7, 1.3),
    "damping_scale_range": (0.6, 1.4), "gain_scale_range": (0.6, 1.4),
    "delay_range": (1, 4), "goal_range": {'low': [-0.1, -0.1, 0.0], 'high': [0.1, 0.1, 0.0]},
}

CURRICULUM_3_MEDIUM = {
    "mass_range": (0.5, 1.5), "friction_range": (0.5, 1.5),
    "damping_scale_range": (0.4, 1.6), "gain_scale_range": (0.4, 1.6),
    "delay_range": (1, 6), "goal_range": {'low': [-0.1, -0.1, 0.0], 'high': [0.1, 0.1, 0.0]},
}

CURRICULUM_4_MEDIUM_PLUS = {
    "mass_range": (0.2, 2.0), "friction_range": (0.2, 2.0),
    "damping_scale_range": (0.2, 2.0), "gain_scale_range": (0.2, 5.0),
    "delay_range": (1, 8), "goal_range": {'low': [-0.15, -0.15, 0.0], 'high': [0.15, 0.15, 0.0]},
}

CURRICULUM_5_HARD = {
    "mass_range": (0.05, 3.0), "friction_range": (0.05, 3.0),
    "damping_scale_range": (0.001, 0.3), "gain_scale_range": (0.0, 10.0),
    "delay_range": (1, 10), "goal_range": {'low': [-0.15, -0.15, 0.0], 'high': [0.15, 0.15, 0.0]},
}

# 为5级课程设计的更平滑的时间表
CURRICULUM_SCHEDULE = {
    400_000: CURRICULUM_2_EASY_PLUS,
    800_000: CURRICULUM_3_MEDIUM,
    1_200_000: CURRICULUM_4_MEDIUM_PLUS,
    1_600_000: CURRICULUM_5_HARD,
}

# (CurriculumCallback 类的代码保持不变)
class CurriculumCallback(BaseCallback):
    def __init__(self, schedule, verbose=0):
        super().__init__(verbose)
        self.schedule = sorted(schedule.items())
        self.stage_index = 0
    def _on_step(self) -> bool:
        if self.stage_index >= len(self.schedule): return True
        next_milestone, new_params = self.schedule[self.stage_index]
        if self.num_timesteps >= next_milestone:
            print(f"\n\n{'*'*20} CURRICULUM UPDATE at Timestep {self.num_timesteps} {'*'*20}")
            print(f"Switching to new randomization ranges for stage {self.stage_index + 2}")
            for key, value in new_params.items():
                self.training_env.set_attr(key, value)
                print(f"  -> Set {key} to {value}")
            self.stage_index += 1
        return True

def main():
    for seed in SEEDS_TO_TRAIN:
        print(f"\n{'='*25} 开始正式训练 Seed: {seed} {'='*25}\n")
        log_dir = f"./logs/dr_model_curriculum_v2_seed{seed}" # 使用新文件夹名
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始环境使用最简单的课程
        env_kwargs = dict(reward_type='sparse', **CURRICULUM_1_EASY, max_episode_steps=100)
        print(f"--- 正在创建 {N_ENVS} 个并行的训练环境 (SubprocVecEnv)... ---")
        env = make_vec_env(
            DomainRandomizationWrapper, n_envs=N_ENVS, seed=seed,
            vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs
        )
        model_params = {
            "policy": "MultiInputPolicy", "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs": dict(n_sampled_goal=4, goal_selection_strategy="future"),
            "verbose": 1, "batch_size": 512, "learning_starts": LEARNING_STARTS,
            "tensorboard_log": f"{log_dir}/tensorboard/"
        }
        print("--- 正在初始化TQC模型 ---")
        model = TQC(env=env, **model_params)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(SAVE_FREQ // N_ENVS, 1), save_path=log_dir, name_prefix="rl_model"
        )
        curriculum_callback = CurriculumCallback(schedule=CURRICULUM_SCHEDULE)
        print(f"--- 开始训练，总步数: {TOTAL_TIMESTEPS} ---")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, curriculum_callback], 
            progress_bar=True
        )
        final_model_path = os.path.join(log_dir, "final_model.zip")
        model.save(final_model_path)
        env.close()
        print(f"\n🎉 Seed {seed} 训练完成！最终模型已保存至: {final_model_path}\n")

if __name__ == "__main__":
    main()