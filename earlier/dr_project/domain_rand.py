import numpy as np
import os
from panda_mujoco_gym.envs.push import FrankaPushEnv

ABS_LOG_PATH = os.path.abspath("domain_rand_log.txt")

class DomainRandomizationWrapper(FrankaPushEnv):
    def __init__(self,
                 reward_type='sparse',
                 goal_range={'low': [-0.1, -0.1, 0.0], 'high': [0.1, 0.1, 0.0]},
                 mass_range=(0.5, 2.0),
                 friction_range=(0.2, 2.0),
                 damping_range=(0.0, 0.1),
                 delay_range=(0, 6),
                 max_episode_steps=50):
        super().__init__(reward_type=reward_type)
        self.goal_range = goal_range
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.damping_range = damping_range
        self.delay_range = delay_range
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._randomized = False

    def _randomize(self):
        mass = np.random.uniform(*self.mass_range)
        friction = np.random.uniform(*self.friction_range)
        damping = np.random.uniform(*self.damping_range)
        delay_steps = np.random.randint(*self.delay_range)
        goal_noise = np.random.uniform(self.goal_range['low'], self.goal_range['high'])
        self.model.body_mass[1] = mass
        self.model.geom_friction[1, 0] = friction
        self.model.dof_damping[:] = damping
        self.delay_steps = delay_steps
        self.goal += goal_noise
        self._randomized = True
        try:
            with open(ABS_LOG_PATH, "a") as f:
                f.write(
                    f"[DomainRand] mass={mass:.3f} friction={friction:.3f} damping={damping:.3f} "
                    f"delay_steps={delay_steps} goal_noise={goal_noise}\n"
                )
                f.flush()
                os.fsync(f.fileno())
            print(f"[domain_rand.py] domain randomized, mass={mass:.3f}, friction={friction:.3f}, delay={delay_steps}")
        except Exception as e:
            print(f"[domain_rand.py] Failed to write log: {e}")

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self._randomized = False
        obs, info = super().reset(**kwargs)
        self._randomize()
        return obs, info

    def step(self, action):
        self._elapsed_steps += 1
        obs, reward, terminated, truncated, info = super().step(action)
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True
        return obs, reward, terminated, truncated, info
