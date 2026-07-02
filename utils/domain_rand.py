# 文件名: domain_rand.py
import numpy as np
import os
import mujoco
from collections import deque
from panda_mujoco_gym.envs.push import FrankaPushEnv

class DomainRandomizationWrapper(FrankaPushEnv):
    def __init__(self,
                 reward_type='sparse',
                 goal_range={'low': [-0.15, -0.15, 0.0], 'high': [0.15, 0.15, 0.0]},
                 mass_range=(0.05, 3.0),
                 friction_range=(0.05, 3.0),
                 damping_scale_range=(0.001, 0.3),
                 gain_scale_range=(0.0, 10.0),
                 delay_range=(1, 10), # 延迟范围 (包含1, 不含10, 即1-9步)
                 max_episode_steps=100):
        
        super().__init__(reward_type=reward_type)
        self.goal_range = goal_range
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.damping_scale_range = damping_scale_range
        self.gain_scale_range = gain_scale_range
        self.delay_range = delay_range
        self.max_episode_steps = max_episode_steps
        
        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'obj')
        self.obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obj_geom')
        
        self.original_damping = self.model.dof_damping.copy()
        self.original_gain = self.model.actuator_gainprm.copy()

        # 使用一个固定大小的列表作为缓冲区，以便随机访问
        self.max_delay = delay_range[1]
        self.action_buffer = [np.zeros(self.action_space.shape) for _ in range(self.max_delay)]
        
        self._elapsed_steps = 0
        self.last_params = {}

    def _randomize_all_params(self):
        """在一个方法内完成所有随机化"""
        mass = np.random.uniform(*self.mass_range)
        friction = np.random.uniform(*self.friction_range)
        damping_scale = np.random.uniform(*self.damping_scale_range)
        gain_scale = np.random.uniform(*self.gain_scale_range)
        delay_steps = np.random.randint(*self.delay_range)
        goal_noise = np.random.uniform(self.goal_range['low'], self.goal_range['high'])
        
        self.model.body_mass[self.obj_body_id] = mass
        self.model.geom_friction[self.obj_geom_id, 0] = friction
        self.model.dof_damping[:] = self.original_damping * damping_scale
        self.model.actuator_gainprm[:, 0] = self.original_gain[:, 0] * gain_scale
        
        super()._sample_goal() 
        self.goal += goal_noise
        
        self.last_params = {
            "mass": mass, "friction": friction, "damping_scale": damping_scale,
            "gain_scale": gain_scale, "delay_steps": delay_steps, "goal_noise": goal_noise
        }

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self.action_buffer = [np.zeros(self.action_space.shape) for _ in range(self.max_delay)]
        obs, info = super().reset(**kwargs)
        self._randomize_all_params()
        return obs, info

    def step(self, action):
        """关键修正：正确实现可变动作延迟"""
        current_delay = self.last_params.get("delay_steps", 1)
        
        # 延迟d步，执行倒数第d个动作, 即 action_buffer[-d]
        action_to_execute = self.action_buffer[-current_delay]
        
        # 将当前新动作追加到列表末尾，并移除最旧的动作
        self.action_buffer.append(action.copy())
        self.action_buffer.pop(0)
        
        obs, reward, terminated, truncated, info = super().step(action_to_execute)
        
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True
        
        if "is_success" not in info:
            info["is_success"] = (reward == 0)

        return obs, reward, terminated, truncated, info
