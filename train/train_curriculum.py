# filename: train_curriculum.py
# (This is the full, corrected script)

import os
import time
import argparse
import numpy as np
import gymnasium as gym

from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback # <-- 添加这一行
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.her import HerReplayBuffer # <-- 添加这一行

import utils.domain_rand as domain_rand

# FIX 1: Add necessary imports for the corrected __init__ method
import mujoco
from panda_mujoco_gym.envs.push import FrankaPushEnv

# ==============================================================================
# 1. OUR CUSTOM WRAPPER (CORRECTED AND COMPLETE)
# ==============================================================================
class CoupledDomainRandomizationWrapper(domain_rand.DomainRandomizationWrapper):
    """
    This is the corrected version of the wrapper.
    It manually performs the initialization of its parent and grandparent classes
    to correctly handle all arguments without modifying the parent files.
    """
    def __init__(self, **kwargs):
        # FIX 2: Manually handle initialization to prevent TypeError.
        # 1. Manually initialize the grandparent (FrankaPushEnv) with its expected arguments.
        # This bypasses the problematic parent __init__ and correctly sets render_mode.
        franka_kwargs = {
            'reward_type': kwargs.get('reward_type', 'sparse'),
            'render_mode': kwargs.get('render_mode', 'rgb_array')
        }
        FrankaPushEnv.__init__(self, **franka_kwargs)

        # 2. Manually replicate the initialization logic from DomainRandomizationWrapper.
        # This sets up the domain randomization parameters correctly.
        # We get the curriculum values directly from kwargs.
        self.goal_range = kwargs.get('goal_range', {'low': [-0.15, -0.15, 0.0], 'high': [0.15, 0.15, 0.0]})
        self.mass_range = kwargs.get('mass_range')
        self.friction_range = kwargs.get('friction_range')
        self.damping_scale_range = kwargs.get('damping_scale_range')
        self.gain_scale_range = kwargs.get('gain_scale_range')
        self.delay_range = kwargs.get('delay_range')
        self.max_episode_steps = kwargs.get('max_episode_steps', 100)

        # Copying necessary attributes setup from DomainRandomizationWrapper
        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'obj')
        self.obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obj_geom')
        self.original_damping = self.model.dof_damping.copy()
        self.original_gain = self.model.actuator_gainprm.copy()
        self.max_delay = self.delay_range[1]
        self.action_buffer = [np.zeros(self.action_space.shape) for _ in range(self.max_delay)]
        self._elapsed_steps = 0
        self.last_params = {}

        # 3. Finally, add the child-specific logic.
        self.zeta_range = (0.7, 1.3)
        print(f"--- [Custom Wrapper] Initialized successfully with manual setup. ---")

# 在 train_curriculum.py 文件中
# 找到 CoupledDomainRandomizationWrapper 类

    def _randomize_all_params(self): # FIX 3: Correctly named method to override parent
        """
        [重要修正]
        实现完整的分层耦合随机化策略，使训练环境与测试环境的物理规律保持一致。
        """
        # --- 第一层：物理任务的耦合约束 (确保可行性) ---
        while True:
            mass = np.random.uniform(*self.mass_range)
            friction = np.random.uniform(*self.friction_range)
            # 物理耦合约束：确保推力可行
            if mass * friction <= 4.0:
                break
        
        self.model.body_mass[self.obj_body_id] = mass
        self.model.geom_friction[self.obj_geom_id, 0] = friction

        # --- 第二层：控制策略的耦合约束 (确保稳定性) ---
        # 使用拒绝采样，直到找到满足所有耦合条件的参数组合
        while True:
            # 1. 随机采样一个增益/刚度 K
            gain_scale = np.random.uniform(*self.gain_scale_range)

            # 2. 随机采样一个合理的阻尼比 zeta
            zeta = np.random.uniform(*self.zeta_range) # self.zeta_range = (0.7, 1.3)

            # 3. 根据 K 和 zeta 计算出对应的阻尼 D
            damping_scale = zeta * 2 * np.sqrt(gain_scale)

            # 4. 检查计算出的 damping_scale 是否在当前课程阶段的期望范围内
            # 这是将课程学习与物理耦合完美结合的关键点
            if self.damping_scale_range[0] <= damping_scale <= self.damping_scale_range[1]:
                break # 找到有效组合，跳出循环

        # 将最终满足条件的参数应用到模型中
        self.model.actuator_gainprm[:, 0] = self.original_gain[:, 0] * gain_scale
        self.model.dof_damping[:] = self.original_damping * damping_scale

        # --- 其他独立随机化 ---
        delay_steps = np.random.randint(*self.delay_range)
        goal_noise = np.random.uniform(self.goal_range['low'], self.goal_range['high'])
        
        super()._sample_goal() 
        self.goal += goal_noise
        
        # 记录所有最终生效的参数，以便分析
        self.last_params = {
            "mass": mass, "friction": friction, "damping_scale": damping_scale,
            "gain_scale": gain_scale, "delay_steps": delay_steps, "goal_noise": goal_noise,
            "zeta": zeta # 额外记录阻尼比
        }

# ==============================================================================
# 2. CURRICULUM STAGES (Unchanged)
# ==============================================================================
CURRICULUM_PARAMS = {
    1: {"name": "Stage 1: Basics", "params": {"delay_range": (1, 2), "mass_range": (1.5, 2.0), "damping_scale_range": (4.0, 5.0), "friction_range": (0.6, 0.9), "gain_scale_range": (4.0, 6.0)}},
    2: {"name": "Stage 2: Robust Domain", "params": {"delay_range": (1, 2), "mass_range": (0.5, 3.0), "damping_scale_range": (3.0, 5.0), "friction_range": (0.05, 1.5), "gain_scale_range": (0.5, 10.0)}},
    3: {"name": "Stage 3: Full Domain", "params": {"delay_range": (1, 2), "mass_range": (0.05, 3.0), "damping_scale_range": (0.5, 5.0), "friction_range": (0.05, 1.5), "gain_scale_range": (0.5, 10.0)}}
}

# ==============================================================================
# 3. MAIN TRAINING LOGIC (Unchanged)
# ==============================================================================
def main(args):
    """Main training function."""
    if args.stage not in CURRICULUM_PARAMS:
        raise ValueError(f"Invalid stage: {args.stage}. Choose from {list(CURRICULUM_PARAMS.keys())}.")
    
    stage_info = CURRICULUM_PARAMS[args.stage]
    env_params = stage_info["params"]
    print(f"\n{'='*20} Preparing: {stage_info['name']} {'='*20}")
    print(f"Using parameters: {env_params}")

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"stage{args.stage}_seed{args.seed}_{int(time.time())}"
        
    log_dir = os.path.join(args.log_folder, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}")

    # The env_kwargs now contains ALL parameters, which our new __init__ will handle correctly.
    env_kwargs = env_params.copy()
    env_kwargs['reward_type'] = 'sparse' # Ensure reward_type is passed
    env_kwargs['render_mode'] = "human" if args.render else "rgb_array"
    
    env = make_vec_env(CoupledDomainRandomizationWrapper, n_envs=args.n_envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)

    if args.load_model_path:
        print(f"\nLoading pre-trained model from {args.load_model_path}")
        model = TQC.load(args.load_model_path, env=env)
        model.set_logger(configure_logger(log_dir, ["stdout", "csv", "tensorboard"]))
    else:
        print("\nCreating new model from scratch")
        model = TQC(
            "MultiInputPolicy",
            env,
            gamma=0.95,
            learning_rate=1e-4,
            batch_size=512,
            buffer_size=1000000,
            learning_starts=25000,
            tau=0.005,
            policy_kwargs=dict(net_arch=[512, 512, 512]),
            # --- 以下是新增的关键部分 ---
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,  # 这就是 k 值
                goal_selection_strategy="future",
            ),
            target_entropy=-2.0,  # 手动设置目标熵，强制增加探索
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
            verbose=1
        )
    print("\nSetting up checkpoint callback...")
    # 每 50,000 步保存一次模型
    # 注意：save_freq 是基于单个环境的步数，所以需要除以并行环境数
    save_freq = max(10000 // args.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=log_dir,
        name_prefix="rl_model"
    )
    # --------------------------
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, log_interval=4, tb_log_name=run_name,callback=checkpoint_callback,
                reset_num_timesteps=not args.load_model_path, progress_bar=args.progress)

    final_model_path = os.path.join(log_dir, "final_model.zip")
    print(f"\nTraining complete. Saving final model to {final_model_path}")
    model.save(final_model_path)
    
    env.close()

# ==============================================================================
# 4. ARGUMENT PARSING (Unchanged)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TQC agent with a curriculum learning strategy.")
    parser.add_argument("--stage", type=int, required=True, help="Curriculum stage (1, 2, or 3)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed (e.g., 0, 42, 123)")
    parser.add_argument("--total-timesteps", type=int, default=1_500_000, help="Timesteps for this run")
    parser.add_argument("--n-envs", type=int, default=6, help="Number of parallel environments")
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to a pre-trained model")
    parser.add_argument("--log-folder", type=str, default="./training_logs_curriculum", help="Root folder for logs")
    parser.add_argument("--run-name", type=str, default=None, help="Specific name for this run's log folder")
    parser.add_argument("--tensorboard-log", type=str, default="./tensorboard_logs_curriculum", help="Root for TensorBoard logs")
    parser.add_argument("--render", action="store_true", help="Render one environment")
    parser.add_argument("--progress", action="store_true", help="Show a tqdm progress bar")
    
    args = parser.parse_args()
    main(args)