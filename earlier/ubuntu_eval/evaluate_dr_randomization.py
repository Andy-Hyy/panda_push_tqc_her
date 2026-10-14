#!/usr/bin/env python3
import os
import argparse
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TQC
from domain_rand import DomainRandomizationWrapper

def parse_range(s, cast=float):
    """把 'min,max' 字符串解析为 (min, max)"""
    parts = s.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Range 参数格式必须是 'low,high'，但收到: {s}")
    return cast(parts[0]), cast(parts[1])

def parse_goal_range(s):
    """把 'lowx,lowy,lowz;highx,highy,highz' 解析成 {'low': [...], 'high': [...]}"""
    try:
        low_s, high_s = s.split(';')
        low = [float(x) for x in low_s.split(',')]
        high = [float(x) for x in high_s.split(',')]
        if len(low) != len(high):
            raise ValueError
    except Exception:
        raise argparse.ArgumentTypeError(
            "goal-range 格式必须是 'lowx,lowy,lowz;highx,highy,highz'"
        )
    return {"low": low, "high": high}

def main():
    p = argparse.ArgumentParser(
        description="在多参数随机化下评估 DR 模型的鲁棒性"
    )
    p.add_argument("--model-path", type=str, required=True,
                   help="训练后保存的模型 (.zip) 文件路径")
    p.add_argument("--mass-range",   type=parse_range, default=None,
                   help="object_mass 范围，例如 '0.2,3.0'")
    p.add_argument("--friction-range", type=parse_range, default=None,
                   help="object_friction 范围，例如 '0.05,2.5'")
    p.add_argument("--damping-range",  type=parse_range, default=None,
                   help="object_damping 范围，例如 '0.01,0.3'")
    # 这里去掉了 cast=int
    p.add_argument("--delay-range", type=parse_range, default=None,
                   help="delay_range 范围 (整数)，例如 '0,10'")
    p.add_argument("--goal-range", type=parse_goal_range, default=None,
                   help="goal_range 范围，例如 '-0.15,-0.15,0.0;0.15,0.15,0.0'")
    p.add_argument("--episodes",   type=int, default=200,
                   help="评估总回合数 (默认为200)")
    args = p.parse_args()

    # 组装 DR 配置
    dr_kwargs = {"reward_type": "sparse"}
    if args.mass_range:
        dr_kwargs["mass_range"] = args.mass_range
    if args.friction_range:
        dr_kwargs["friction_range"] = args.friction_range
    if args.damping_range:
        dr_kwargs["damping_range"] = args.damping_range
    if args.delay_range:
        dr_kwargs["delay_range"] = tuple(args.delay_range)
    if args.goal_range:
        dr_kwargs["goal_range"] = args.goal_range

    print("=== DR 测试配置 ===")
    for k, v in dr_kwargs.items():
        print(f"  {k}: {v}")
    print("===================")

    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = TQC.load(args.model_path)

    successes = 0
    for ep in range(1, args.episodes + 1):
        # 每回合新建环境以重新随机化参数
        env = DomainRandomizationWrapper(**dr_kwargs)
        env = Monitor(env)
        obs = env.reset()

        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        if info.get("is_success", False):
            successes += 1

        if ep % 20 == 0 or ep == args.episodes:
            print(f"Episode {ep}/{args.episodes}: current success_rate = {successes/ep*100:.2f}%")

    final_rate = successes / args.episodes * 100
    print(f"\n=== 最终鲁棒性评估结果 ===")
    print(f"Evaluated Episodes: {args.episodes}")
    print(f"Overall Success Rate: {final_rate:.2f}%")
    print("===========================")

if __name__ == "__main__":
    main()
