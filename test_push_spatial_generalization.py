import gymnasium as gym
from sb3_contrib import TQC
import numpy as np
import argparse
import csv
from panda_mujoco_gym.envs.push import FrankaPushEnv

# 随机性环境列表
GOAL_XY_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] 
SEED_LIST = [0, 42, 123]
NUM_EPISODES = 200
MAX_EPISODE_STEPS = 100

def evaluate_model(model_path, out_csv="eval_results.csv"):
    results = []
    
    # 步骤1: 创建唯一的、贯穿整个评测过程的环境实例
    # 使用列表中的第一个值进行初始化
    print("[Configuration] 正在创建唯一的持久化环境...")
    env = FrankaPushEnv(
        reward_type='sparse',
        goal_xy_range=GOAL_XY_LIST[0],
    )
    
    # 步骤2: 加载模型并与这个唯一的环境关联
    print(f"[Configuration] 正在从 {model_path} 加载模型...")
    model = TQC.load(model_path, env=env)

    try:
        # 步骤3: 循环遍历不同的目标配置
        for goal_xy in GOAL_XY_LIST:
            print(f"\n[Configuration] 正在为 goal_xy_range = {goal_xy:.2f} 修改环境参数...")
            
            # 步骤4: 直接修改现有环境实例的内部参数
            # 这是最关键的一步，我们必须模拟 __init__ 方法中的完整逻辑
            env.goal_xy_range = goal_xy
            
            # 同样，从 `panda_env.py` 的 __init__ 中我们知道，需要重新计算这些依赖值
            # 并且需要加上写死在代码里的 0.6 偏移量
            env.goal_range_low = np.array([-env.goal_xy_range / 2 + env.goal_x_offset, -env.goal_xy_range / 2, 0])
            env.goal_range_high = np.array([env.goal_xy_range / 2 + env.goal_x_offset, env.goal_xy_range / 2, env.goal_z_range])
            env.goal_range_low[0] += 0.6
            env.goal_range_high[0] += 0.6
            
            # 内层循环，遍历不同的seed
            for seed in SEED_LIST:
                print(f"评测: goal_xy={goal_xy:.2f}, seed={seed}")
                successes = 0
                for i in range(NUM_EPISODES):
                    # 每次重置环境，它会使用我们刚刚更新过的参数来采样新的目标
                    obs, _ = env.reset(seed=seed + i)
                    terminated = False
                    truncated = False
                    episode_steps = 0
                    
                    while not (terminated or truncated):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_steps += 1
                        if episode_steps >= MAX_EPISODE_STEPS:
                            truncated = True
                    if info.get('is_success'):
                        successes += 1
                
                fail = NUM_EPISODES - successes
                rate = successes / NUM_EPISODES
                print(f"  -> 成功次数: {successes}, 失败次数: {fail}, 成功率: {rate:.3f}")
                results.append({
                    "goal_xy": goal_xy,
                    "seed": seed,
                    "success": successes,
                    "fail": fail,
                    "success_rate": rate
                })
    finally:
        # 步骤5: 在所有评测任务结束后，关闭这个唯一的环境实例
        print("\n[Cleanup] 所有评测完成，正在关闭唯一的环境...")
        env.close()

    # 将所有结果保存到CSV文件
    with open(out_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["goal_xy", "seed", "success", "fail", "success_rate"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n✅ 已保存评测结果至 {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="TQC模型路径")
    parser.add_argument("--output", type=str, default="eval_results.csv", help="输出csv文件名")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.output)