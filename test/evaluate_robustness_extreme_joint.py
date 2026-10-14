import gymnasium as gym
from sb3_contrib import TQC
import numpy as np
import pandas as pd
import argparse
import itertools

try:
    import panda_mujoco_gym
except ImportError:
    print("错误：无法导入 'panda_mujoco_gym'。")
    exit()

# 你可以在这里自定义 sweep 的极端参数和取值
EXTREME_PARAM_GRID = {
    'object_mass': [0.05, 3.0],           # 机械臂能力极限
    'object_friction': [0.05, 4],      # 极端摩擦
    'object_damping': [0.01, 0.3],       # 极端阻尼
    'actuator_gain': [50, 150],        # 可加
    #'object_delay': [0, 10],           # 若你有 delay 支持
    'goal_xy_range': [0.2, 0.4],       # 若 goal 支持极端扰动
}

def evaluate_joint_extreme(
    model_path: str,
    param_grid: dict,
    num_episodes: int
):
    print("--- 多参数极端 sweep 测试 ---")
    print("模型路径:", model_path)
    print("联合 sweep 参数:")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
    print(f"每组极端参数测试回合数: {num_episodes}")
    print()

    env = None
    try:
        env = gym.make("FrankaPushSparse-v0")
        model = TQC.load(model_path, env=env)
        print("✅ 环境和模型创建成功。\n")
    except Exception as e:
        print(f"❌ 环境或模型初始化失败: {e}")
        if env:
            env.close()
        return

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    all_combinations = list(itertools.product(*value_lists))

    results = []
    for values in all_combinations:
        param_combo = dict(zip(keys, values))
        print(f"--- 测试参数组: {param_combo} ---")
        # 组装 set_physics_properties 的参数
        kwargs_to_set = {}
        for k, v in param_combo.items():
            if k == 'object_friction':
                kwargs_to_set[k] = [v, 0.005, 0.0001]
            else:
                kwargs_to_set[k] = v
        env.unwrapped.set_physics_properties(**kwargs_to_set)

        successes = 0
        for i in range(num_episodes):
            obs, _ = env.reset()
            terminated, truncated = False, False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
            if info.get('is_success'):
                successes += 1
        success_rate = successes / num_episodes
        combo_str = ', '.join([f"{k}={v}" for k, v in param_combo.items()])
        print(f"  > 成功率: {success_rate:.2%}\n")
        results.append({'params': combo_str, 'success_rate': success_rate})

    env.close()
    print("--- sweep 评估完成 ---\n")
    results_df = pd.DataFrame(results)
    pd.set_option('display.precision', 3)
    pd.set_option('display.float_format', '{:.2%}'.format)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多参数极端区间联合 sweep 评估 DR 模型鲁棒性。")
    parser.add_argument("--model-path", type=str, required=True, help="指向已保存的.zip模型文件的路径。")
    parser.add_argument("--episodes", type=int, default=50, help="每组参数下测试回合数。")
    args = parser.parse_args()

    evaluate_joint_extreme(
        model_path=args.model_path,
        param_grid=EXTREME_PARAM_GRID,
        num_episodes=args.episodes
    )
