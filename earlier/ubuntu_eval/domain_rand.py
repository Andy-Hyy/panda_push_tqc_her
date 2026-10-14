# domain_rand.py

import os
import gymnasium as gym
# 确保 Push 环境已经注册
import panda_mujoco_gym.envs.push  # noqa: F401

class DomainRandomizationWrapper(gym.Wrapper):
    """
    每次 reset 时对物理参数做域随机化，然后返回带随机化配置的 Push 环境。
    """

    def __init__(
        self,
        reward_type: str = "sparse",
        mass_range: tuple[float, float] = None,
        friction_range: tuple[float, float] = None,
        damping_range: tuple[float, float] = None,
        delay_range: tuple[int, int]    = None,
        goal_range: dict                = None,
    ):
        # 避免 macOS 下 OpenMP 重复加载导致的崩溃
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # 组装 DR 配置
        dr_config: dict = {}
        if mass_range     is not None: dr_config["mass_range"]     = mass_range
        if friction_range is not None: dr_config["friction_range"] = friction_range
        if damping_range  is not None: dr_config["damping_range"]  = damping_range
        if delay_range    is not None: dr_config["delay_range"]    = delay_range
        if goal_range     is not None: dr_config["goal_range"]     = goal_range

        # 创建带域随机化的 Push 环境
        env = gym.make(
            "FrankaPushSparse-v0",
            reward_type=reward_type,
            domain_randomization=True,
            dr_config=dr_config,
        )
        super().__init__(env)

    def reset(self, **kwargs):
        # 直接调用底层环境的 reset
        return self.env.reset(**kwargs)

    def step(self, action):
        # 直接调用底层环境的 step
        return self.env.step(action)
