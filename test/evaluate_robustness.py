# evaluate_robustness.py (FINAL FULL-FEATURED VERSION)

import gymnasium as gym
from sb3_contrib import TQC
import numpy as np
import pandas as pd
import argparse

try:
    import panda_mujoco_gym
except ImportError:
    print("错误：无法导入 'panda_mujoco_gym'。")
    print("请确保您已经通过 'pip install -e .' 正确安装了您的自定义环境包。")
    exit()

def evaluate_robustness(
    model_path: str,
    param_name: str,
    param_min: float,
    param_max: float,
    param_steps: int,
    num_episodes: int
):
    """
    通过重用同一个环境实例来评估模型的鲁棒性，支持多种参数。
    """
    print("--- 开始模型鲁棒性评估 (全功能架构) ---")
    print(f"模型路径: {model_path}")
    print(f"评估参数: '{param_name}'")
    
    param_values = np.linspace(param_min, param_max, param_steps)
    print(f"参数范围: 从 {param_min} 到 {param_max}，共 {param_steps} 个点。")
    print(f"每个点的评估回合数: {num_episodes}\n")

    env = None
    try:
        print("正在创建环境和加载模型...")
        env = gym.make("FrankaPushSparse-v0")
        model = TQC.load(model_path, env=env)
        print("✅ 环境和模型创建成功。\n")
    except Exception as e:
        print(f"❌ 环境或模型初始化失败: {e}")
        if env:
            env.close()
        return

    results = []
    for value in param_values:
        print(f"--- 正在测试 {param_name} = {value:.3f} ---")
        
        try:
            # ===> 扩展的逻辑，用于处理所有新参数 <===
            kwargs_to_set = {}
            if param_name == 'object_friction':
                kwargs_to_set = {'object_friction': [value, 0.005, 0.0001]}
            elif param_name == 'object_mass':
                kwargs_to_set = {'object_mass': value}
            elif param_name == 'xy_range':
                kwargs_to_set = {'obj_xy_range': value, 'goal_xy_range': value}
            elif param_name == 'object_damping':
                kwargs_to_set = {'object_damping': value}
            elif param_name == 'actuator_gain':
                kwargs_to_set = {'actuator_gain': value}
            else:
                print(f"警告：未知的参数名称 '{param_name}'，将不修改任何参数。")
            
            if kwargs_to_set:
                env.unwrapped.set_physics_properties(**kwargs_to_set)
            # =================================================

            successes = 0
            for i in range(num_episodes):
                obs, _ = env.reset()
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                if info.get('is_success'):
                    successes += 1
            
            success_rate = successes / num_episodes
            results.append({'param_value': value, 'success_rate': success_rate})
            print(f"  > 成功率: {success_rate:.2%}\n")

        except Exception as e:
            print(f"❌ 在处理参数 {param_name}={value:.3f} 时发生错误: {e}")
            continue

    env.close()

    print("--- 评估完成 ---")
    if not results:
        print("评估过程中没有成功的回合，无法生成报告。请检查错误日志。")
        return

    results_df = pd.DataFrame(results)
    pd.set_option('display.precision', 3)
    pd.set_option('display.float_format', '{:.2%}'.format)
    print("最终鲁棒性报告:")
    results_df_display = results_df.copy()
    results_df_display['param_value'] = results_df_display['param_value'].map('{:.3f}'.format)
    print(results_df_display.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估一个已训练的Panda Push模型在一系列参数下的鲁棒性。")
    parser.add_argument("--model-path", type=str, required=True, help="指向已保存的.zip模型文件的路径。")
    parser.add_argument("--param-name", type=str, required=True, help="要测试的环境参数名称 (例如 'object_mass', 'object_friction', 'xy_range', 'object_damping', 'actuator_gain')。")
    parser.add_argument("--param-min", type=float, default=0.1, help="参数范围的最小值。")
    parser.add_argument("--param-max", type=float, default=0.8, help="参数范围的最大值。")
    parser.add_argument("--param-steps", type=int, default=8, help="在范围内均勻取点的数量。")
    parser.add_argument("--episodes", type=int, default=50, help="每个参数点上用于评估的总回合数。")
    
    args = parser.parse_args()

    evaluate_robustness(
        model_path=args.model_path,
        param_name=args.param_name,
        param_min=args.param_min,
        param_max=args.param_max,
        param_steps=args.param_steps,
        num_episodes=args.episodes
    )