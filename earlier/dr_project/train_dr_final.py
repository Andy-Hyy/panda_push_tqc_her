# train_dr_final.py

import os
import time
import argparse

from sb3_contrib import TQC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from domain_rand import DomainRandomizationWrapper

def make_env(seed):
    def _init():
        env = DomainRandomizationWrapper(
            reward_type='sparse',
            goal_range={'low': [-0.1, -0.1, 0.0], 'high': [0.1, 0.1, 0.0]},
            mass_range=(0.5, 2.0),
            friction_range=(0.2, 2.0),
            damping_range=(0.0, 0.1),
            delay_range=(0, 6)
        )
        env.reset(seed=seed)
        return env
    return _init

def train(seed=0, batch_size=512, n_sampled_goal=4, total_timesteps=1_000_000, n_envs=2):
    print(f"--- 域随机化训练（DummyVecEnv, n_envs={n_envs}）---")
    print(f"seed={seed}, batch_size={batch_size}, n_sampled_goal={n_sampled_goal}")

    # VecEnv 环境
    env = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])

    # 日志路径
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

    # 模型
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
    parser = argparse.ArgumentParser(description="域随机化 FrankaPush TQC+HER 训练")
    parser.add_argument("--seed", type=int, default=0, help="训练使用的随机种子")
    parser.add_argument("--batch_size", type=int, default=512, help="批量大小")
    parser.add_argument("--n_sampled_goal", type=int, default=4, help="HER采样数")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="总训练步数")
    parser.add_argument("--n_envs", type=int, default=2, help="并行环境数")
    args = parser.parse_args()

    train(
        seed=args.seed,
        batch_size=args.batch_size,
        n_sampled_goal=args.n_sampled_goal,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs
    )
