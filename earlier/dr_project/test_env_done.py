from domain_rand import DomainRandomizationWrapper

# 用你平时训练用的参数
env = DomainRandomizationWrapper(
    reward_type='sparse',
    goal_range={'low': [-0.1, -0.1, 0.0], 'high': [0.1, 0.1, 0.0]},
    mass_range=(0.5, 2.0),
    friction_range=(0.2, 2.0),
    damping_range=(0.0, 0.1),
    delay_range=(0, 6)
)

obs, info = env.reset()
step_count = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step {step_count}: terminated={terminated}, truncated={truncated}")
    if terminated or truncated:
        print("Episode ended at step", step_count)
        break
    step_count += 1
