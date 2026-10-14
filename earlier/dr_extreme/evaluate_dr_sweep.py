import argparse
import numpy as np
from sb3_contrib import TQC
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

def parse_range(val):
    # 例如 "0.05,2.5" -> (0.05, 2.5)
    return tuple(float(x) for x in val.split(","))

def run_sweep(model_path, env_id, sweep_param, sweep_range, episodes, other_kwargs):
    # 采样 sweep_range
    sweep_points = np.linspace(sweep_range[0], sweep_range[1], num=10)
    results = []

    for param_val in sweep_points:
        # 更新 domain randomization 配置
        dr_config = {sweep_param: (param_val, param_val)}  # 固定 sweep 参数，其它参数用默认/other_kwargs

        # 组装所有 domain randomization 参数
        env_kwargs = {
            "reward_type": "sparse",
            "domain_randomization": True,
            "dr_config": dr_config,
            **other_kwargs
        }

        # 创建环境和模型
        def make_env():
            return gym.make(env_id, **env_kwargs)
        env = DummyVecEnv([make_env])

        print(f"\n--- 正在测试 {sweep_param} = {param_val:.3f} ---")
        model = TQC.load(model_path, env=env)

        # rollout
        success_count = 0
        for ep in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if isinstance(done, (list, np.ndarray)):
                    done = done[0]
                if done and (isinstance(info, list) and info[0].get('is_success', 0) or info.get('is_success', 0)):
                    if isinstance(info, list):
                        success_count += info[0].get('is_success', 0)
                    else:
                        success_count += info.get('is_success', 0)
        success_rate = success_count / episodes * 100
        print(f"  > 成功率: {success_rate:.2f}%")
        results.append((param_val, success_rate))

    # 输出 sweep 总表格
    print("\n--- sweep 结果 ---")
    print("param_value  success_rate")
    for pv, sr in results:
        print(f"{pv:10.3f}  {sr:8.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-id", default="FrankaPushSparse-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--sweep-param", type=str, choices=["object_friction", "object_mass", "object_damping", "object_delay"], required=True)
    parser.add_argument("--sweep-range", type=str, required=True, help="例如 0.05,2.5")
    # 可扩展：其它默认 domain randomization 参数
    args = parser.parse_args()

    # 解析 sweep 范围
    sweep_range = parse_range(args.sweep_range)
    run_sweep(
        model_path=args.model_path,
        env_id=args.env_id,
        sweep_param=args.sweep_param,
        sweep_range=sweep_range,
        episodes=args.episodes,
        other_kwargs={}
    )
