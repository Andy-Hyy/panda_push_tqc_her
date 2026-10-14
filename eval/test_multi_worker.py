# 文件名: test_multi_worker.py
# (版本已修改，采纳了所有优化建议)

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "utils")))
os.chdir(_HERE)

import numpy as np
import random
import torch
import pandas as pd
import argparse
from sb3_contrib import TQC
import panda_mujoco_gym
from domain_rand import DomainRandomizationWrapper # 依然需要导入基类

# --- 常量定义 ---
SEEDS = [0, 42, 123]
N_EPISODES_PER_SEED = 200
MAX_EPISODE_STEPS = 100

# --- 核心修改：定义一个继承自原Wrapper的新类，以实现耦合随机化 ---
class CoupledDomainRandomizationWrapper(DomainRandomizationWrapper):
    """
    一个实现了分层耦合随机化的新Wrapper。
    1. 物理参数耦合：确保推力在可持续范围内。
    2. 控制参数耦合：确保控制器行为稳定、合理。
    """
    def __init__(self, **kwargs):
        # 使用更合理的默认范围初始化父类
        super().__init__(
            mass_range=(0.5, 3.0),
            friction_range=(0.05, 1.5), # 修正了摩擦系数的物理现实范围
            damping_scale_range=(6.5, 8.0), # 这是一个占位符，实际值将被耦合计算覆盖，修改，从0.5修改为3.0
            gain_scale_range=(0.5, 10.0), # 修正了增益范围，排除了0
            **kwargs
        )
        print("[Wrapper] 已初始化耦合域随机化环境。")

    def _randomize_all_params(self):
        """
        重写此方法以实现分层耦合随机化。
        """
        # --- 第一层：物理任务的耦合约束 ---
        while True:
            mass = np.random.uniform(*self.mass_range)
            friction = np.random.uniform(*self.friction_range)
            # 物理耦合约束：确保推力可行
            if mass * friction <= 4.0:
                break
        
        self.model.body_mass[self.obj_body_id] = mass
        self.model.geom_friction[self.obj_geom_id, 0] = friction

        # --- 第二层：控制策略的耦合约束 (含范围控制) ---
        # 定义 damping_scale 的期望范围
        DAMPING_SCALE_MIN = 6.5
        DAMPING_SCALE_MAX = 8.0

        # 使用拒绝采样，直到找到满足所有耦合条件的参数组合
        while True:
            # 1. 随机采样一个增益/刚度 K
            gain_scale = np.random.uniform(*self.gain_scale_range) # K in [0.5, 10.0]

            # 2. 随机采样一个合理的阻尼比 zeta
            zeta = np.random.uniform(0.7, 1.3) # ζ in [0.7, 1.3]

            # 3. 根据 K 和 zeta 计算出对应的阻尼 D
            damping_scale = zeta * 2 * np.sqrt(gain_scale)

            # 4. 检查计算出的 damping_scale 是否在期望范围内
            if DAMPING_SCALE_MIN <= damping_scale <= DAMPING_SCALE_MAX:
                break # 找到有效组合，跳出循环

# 将最终满足条件的参数应用到模型中
        self.model.actuator_gainprm[:, 0] = self.original_gain[:, 0] * gain_scale
        self.model.dof_damping[:] = self.original_damping * damping_scale
        # --- 其他扰动 ---
        delay_steps = np.random.randint(*self.delay_range)
        goal_noise = np.random.uniform(self.goal_range['low'], self.goal_range['high'])
        
        super()._sample_goal() 
        self.goal += goal_noise
        
        # 记录所有最终生效的参数，以便分析
        self.last_params = {
            "mass": mass, "friction": friction, "damping_scale": damping_scale,
            "gain_scale": gain_scale, "delay_steps": delay_steps, "goal_noise": goal_noise,
            "zeta": zeta # 额外记录阻尼比，便于后续分析
        }

def run_multi_param_evaluation(model_path, output_csv):
    """对单个模型运行多参数联合扰动测试（使用新的耦合Wrapper）"""
    all_logs = []
    
    print(f"\n[Worker] 开始对模型 {model_path} 进行多参数联合扰动测试...")

    print("[Worker] 正在创建唯一的、持久化的评测环境...")
    env = None
    try:
        # --- 使用我们新定义的、更科学的Wrapper ---
        env = CoupledDomainRandomizationWrapper(
            reward_type='sparse',
            goal_range={'low': [-0.15, -0.15, 0.0], 'high': [0.15, 0.15, 0.0]},
            delay_range=(1, 2),
            max_episode_steps=MAX_EPISODE_STEPS
        )
        
        print("[Worker] 正在加载模型...")
        model = TQC.load(model_path, env=env)
        
        for seed in SEEDS:
            print(f"[Worker] 正在评测 Seed = {seed}...")
            
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            
            for i in range(N_EPISODES_PER_SEED):
                obs, info = env.reset()
                params = env.last_params
                
                terminated, truncated = False, False
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                
                is_success = 1 if info.get("is_success") else 0
                log_entry = {
                    "seed": seed, "episode": i + 1, "mass": params.get("mass"),
                    "friction": params.get("friction"), "damping_scale": params.get("damping_scale"),
                    "gain_scale": params.get("gain_scale"),"delay_steps": params.get("delay_steps"),
                    "goal_noise_x": params.get("goal_noise")[0],
                    "goal_noise_y": params.get("goal_noise")[1],
                    "goal_noise_z": params.get("goal_noise")[2],
                    "zeta": params.get("zeta"), # 记录zeta
                    "success": is_success
                }
                all_logs.append(log_entry)
            
        print(f"[Worker] 模型 {os.path.basename(model_path)} 的所有seed评测完成。")

    finally:
        if env:
            print("[Worker] 正在关闭环境...")
            env.close()

    # 将所有详细日志一次性写入CSV文件
    df = pd.DataFrame(all_logs)
    columns_order = [
        "seed", "episode", "success", "mass", "friction", 
        "damping_scale", "gain_scale", "zeta", "delay_steps",
        "goal_noise_x", "goal_noise_y", "goal_noise_z"
    ]
    # 过滤掉不存在的列以增加灵活性
    df = df[[col for col in columns_order if col in df.columns]]
    df.to_csv(output_csv, index=False)
    print(f"[Worker] 评测结果已保存到 {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Element Generalization Worker (Coupled & Rigorous Version)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()
    run_multi_param_evaluation(args.model_path, args.output_csv)