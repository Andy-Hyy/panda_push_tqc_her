# smoke_test_dr.py
import time
from domain_rand import DomainRandomizationWrapper

if __name__ == "__main__":
    DR_CONFIG = {
        'goal_range':     {'low': [-0.1, -0.1, 0], 'high': [0.1, 0.1, 0]},
        'mass_range':     (0.5, 2.0),
        'friction_range': (0.2, 2.0),
        'damping_range':  (0.0, 0.1),
        'delay_range':    (0, 6),   # 整数步 [0,5]
    }

    # Gymnasium 环境创建
    env = DomainRandomizationWrapper(
        reward_type='sparse',
        **DR_CONFIG
    )

    # 如果想固定随机种子可以：
    # obs, info = env.reset(seed=42)

    for i in range(5):
        obs, info = env.reset()
        print(f"[Reset {i}] params:", env.last_params)
        # 随机采个动作，跑一步
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"  step → reward={reward:.3f}, done={done}")
        time.sleep(0.1)

    env.close()
