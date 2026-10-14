import os
import time
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from domain_rand import DomainRandomizationWrapper

# 修改为你需要的并行数
N_ENVS = 4
TOTAL_TIMESTEPS = 1_000_000
BATCH_SIZE = 512
N_SAMPLED_GOAL = 4
SEED = 0

def make_env(seed):
    def _init():
        env = DomainRandomizationWrapper(
            reward_type='sparse',
            goal_range={'low': [-0.15, -0.15, 0.0], 'high': [0.15, 0.15, 0.0]},
            mass_range=(0.2, 3.0),
            friction_range=(0.05, 2.5),
            damping_range=(0.01, 0.3),
            delay_range=(0, 10)
        )
        env.reset(seed=seed)
        return env
    return _init

def train(seed=SEED, batch_size=BATCH_SIZE, n_sampled_goal=N_SAMPLED_GOAL, total_timesteps=TOTAL_TIMESTEPS, n_envs=N_ENVS):
    print(f"--- 域随机化训练（SubprocVecEnv, n_envs={n_envs}）---")
    print(f"seed={seed}, batch_size={batch_size}, n_sampled_goal={n_sampled_goal}")

    env = SubprocVecEnv([make_env(seed + i) for i in range(n_envs)])

    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    config_name = f"dr_batch{batch_size}_her{n_sampled_goal}_envs{n_envs}"
    log_dir = f"logs/FrankaPushDR_TQC/{config_name}_seed{seed}_{run_timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    final_model_path = os.path.join(log_dir, "final_model.zip")

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(log_dir, "checkpoints/"),
        name_prefix="tqc_frankapush_dr"
    )

    model = TQC(
        policy="MultiInputPolicy",
        env=env,
        seed=seed,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        batch_size=batch_size,
        tau=0.05,
        gamma=0.95,
        top_quantiles_to_drop_per_net=2,
        ent_coef="auto",
        learning_starts=1000,  # 比 n_envs * max_episode_steps 大一些
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
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    model.save(final_model_path)
    print(f"\n训练完成！模型已保存至: {final_model_path}")

if __name__ == "__main__":
    train()
