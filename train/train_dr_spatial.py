import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from sb3_contrib import TQC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

# 假设 panda_mujoco_gym 已经安装并且可以导入
from panda_mujoco_gym.envs.push import FrankaPushEnv

class Float32Wrapper(gym.ObservationWrapper):
    """
    完整版的观测包装器：
    1. 初始化时，将 observation_space 从 float64 转为 float32
    2. 返回观测时，将 observation 数据转为 float32
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        new_spaces = {}
        for key, space in env.observation_space.spaces.items():
            if isinstance(space, spaces.Box) and space.dtype == np.float64:
                new_spaces[key] = spaces.Box(
                    low=space.low.astype(np.float32),
                    high=space.high.astype(np.float32),
                    shape=space.shape,
                    dtype=np.float32
                )
            else:
                new_spaces[key] = space
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, observation):
        return {
            key: value.astype(np.float32) if isinstance(value, np.ndarray) and value.dtype == np.float64 else value
            for key, value in observation.items()
        }

class SimpleRangeRandomizationWrapper(gym.Wrapper):
    """
    随机化 goal_xy_range 参数
    """
    def __init__(self, env: gym.Env, range_dict: dict):
        super().__init__(env)
        assert 'low' in range_dict and 'high' in range_dict
        self.low = range_dict['low']
        self.high = range_dict['high']

    def reset(self, **kwargs):
        unwrapped = self.env.unwrapped
        unwrapped.goal_xy_range = self.np_random.uniform(self.low, self.high)
        return self.env.reset(**kwargs)

# --- 超参数 ---
N_ENVS = 6
TOTAL_STEPS = 500_000
BATCH_SIZE = 512
N_SAMPLED_GOAL = 4
GOAL_RANGE = {'low': 0.3, 'high': 0.8}
MAX_EPISODE_STEPS = 100


def make_env(seed):
    def _init():
        env = FrankaPushEnv(reward_type='sparse')
        env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = Float32Wrapper(env)
        env = SimpleRangeRandomizationWrapper(env, GOAL_RANGE)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(seed):
    device = 'cpu'
    env = SubprocVecEnv([make_env(seed + i) for i in range(N_ENVS)])
    # log 目录
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    cfg = f"dr_spatial_bs{BATCH_SIZE}_her{N_SAMPLED_GOAL}_envs{N_ENVS}"
    log_dir = os.path.join('logs', 'FrankaPushSpatialDR_TQC', f"{cfg}_seed{seed}_{ts}")
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=5000,
        save_path=ckpt_dir,
        name_prefix='tqc_frankapush_spatial_dr'
    )
    cb_list = CallbackList([checkpoint_cb])

    model = TQC(
        policy='MultiInputPolicy',
        env=env,
        seed=seed,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        batch_size=BATCH_SIZE,
        tau=0.05,
        gamma=0.95,
        top_quantiles_to_drop_per_net=2,
        ent_coef='auto',
        learning_starts=10000,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            'n_sampled_goal': N_SAMPLED_GOAL,
            'goal_selection_strategy': 'future'
        },
        policy_kwargs={'net_arch': {'pi': [256]*3, 'qf': [256]*3}, 'n_quantiles': 25},
        tensorboard_log=log_dir,
        device=device,
        verbose=1
    )

    print(f"Starting seed {seed}, logdir: {log_dir}")
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=cb_list,
        progress_bar=True
    )
    final_path = os.path.join(log_dir, 'final_model.zip')
    model.save(final_path)
    print(f"Seed {seed} training done, model saved to: {final_path}")


if __name__ == '__main__':
    for s in [42, 123, 0]:
        train(s)
    print('All seeds done.')
