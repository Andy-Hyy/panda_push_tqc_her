import gymnasium as gym
from sb3_contrib import TQC
import numpy as np
import argparse
import csv
import os
import xml.etree.ElementTree as ET
import tempfile
import re
import shutil
import mujoco

# 关键：导入库以触发gymnasium注册
import panda_mujoco_gym

# --- 常量定义 ---
BASE_PATH = os.path.dirname(__file__)
ASSETS_PATH = os.path.join(BASE_PATH, "panda_mujoco_gym", "assets")
PUSH_XML_PATH = os.path.join(ASSETS_PATH, "push.xml")
MOCAP_XML_PATH = os.path.join(ASSETS_PATH, "panda_mocap.xml")

SEED_LIST = [0, 42, 123]
NUM_EPISODES_PER_SEED = 200
MAX_EPISODE_STEPS = 100

def modify_xml_and_copy_assets(param_name, value, temp_dir):
    """修改XML参数，并将相关的资源文件（meshes）复制到临时目录。"""
    original_mesh_path = os.path.join(ASSETS_PATH, "meshes")
    temp_mesh_path = os.path.join(temp_dir, "meshes")
    if os.path.exists(original_mesh_path):
        shutil.copytree(original_mesh_path, temp_mesh_path)

    tree_push = ET.parse(PUSH_XML_PATH)
    root_push = tree_push.getroot()
    tree_mocap = ET.parse(MOCAP_XML_PATH)
    root_mocap = tree_mocap.getroot()

    if param_name == "object_mass":
        obj_geom = root_push.find(".//geom[@name='obj_geom']")
        if obj_geom is not None: obj_geom.set('mass', str(value))
    elif param_name == "object_friction":
        obj_geom = root_push.find(".//geom[@name='obj_geom']")
        if obj_geom is not None: obj_geom.set('friction', str(value))
    elif param_name == "joint_damping_scale":
        for joint in root_mocap.findall(".//joint"):
            if joint.get('damping'): joint.set('damping', str(1.0 * float(value)))
    elif param_name == "actuator_gain_scale":
        original_gains = {'actuator1': 4500, 'actuator2': 4500, 'actuator3': 3500, 'actuator4': 3500, 'actuator5': 2000, 'actuator6': 2000, 'actuator7': 2000, 'r_gripper_finger_joint': 500, 'l_gripper_finger_joint': 500}
        for actuator in root_mocap.findall(".//actuator/general"):
            act_name = actuator.get('name')
            if act_name in original_gains:
                 new_gain = original_gains[act_name] * float(value)
                 actuator.set('gainprm', str(new_gain))

    temp_push_path = os.path.join(temp_dir, "push_temp.xml")
    temp_mocap_path = os.path.join(temp_dir, "panda_mocap_temp.xml")
    include_element = root_push.find(".//include")
    if include_element is not None:
        include_element.set('file', "panda_mocap_temp.xml")

    tree_push.write(temp_push_path, encoding='utf-8')
    tree_mocap.write(temp_mocap_path, encoding='utf-8')
    return temp_push_path

def run_single_evaluation(model_path, param_name, param_value, output_csv):
    """为单个参数值运行完整的评测流程"""
    with tempfile.TemporaryDirectory() as temp_dir:
        modified_xml_path = modify_xml_and_copy_assets(param_name, param_value, temp_dir)
        
        print(f"\n[Worker] 正在为 '{param_name} = {param_value}' 创建环境...")
        
        eval_env = None
        try:
            # 步骤 1: 使用 gym.make 和修改后的XML创建环境
            # 这个 env 将被用于加载模型
            eval_env = gym.make("FrankaPushSparse-v0", model_path=modified_xml_path)
            
            # 步骤 2: 关键修正！加载模型时必须传入 env
            print(f"[Worker] 正在加载模型...")
            model = TQC.load(model_path, env=eval_env)
            
            total_successes = 0
            for seed in SEED_LIST:
                print(f"[Worker] 正在评测 seed={seed}...")
                for i in range(NUM_EPISODES_PER_SEED):
                    
                    # 步骤 3: 使用 VecEnv 的 API
                    model.get_env().seed(seed + i)
                    obs = model.get_env().reset()

                    episode_steps = 0
                    while True:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, rewards, dones, infos = model.get_env().step(action)
                        episode_steps += 1
                        
                        is_done = dones[0]
                        is_truncated = episode_steps >= MAX_EPISODE_STEPS

                        if is_done or is_truncated:
                            if infos[0].get('is_success'):
                                total_successes += 1
                            break
            
            total_episodes = len(SEED_LIST) * NUM_EPISODES_PER_SEED
            final_rate = total_successes / total_episodes if total_episodes > 0 else 0
            print(f"  -> 参数值: {param_value}, 总成功率: {final_rate:.3f} ({total_successes}/{total_episodes})")

            # 将单次结果追加到CSV文件
            with open(output_csv, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["param_name", "value", "success_rate"])
                writer.writerow({"param_name": param_name, "value": param_value, "success_rate": final_rate})

        finally:
            # VecEnv 是在 model 内部管理的，关闭 model 的 env 即可
            if 'model' in locals() and model.get_env() is not None:
                 print(f"[Worker] 正在关闭环境...")
                 model.get_env().close()
            elif eval_env: # 以防模型加载失败
                 eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Worker")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--param-name", type=str, required=True)
    parser.add_argument("--param-value", required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()

    run_single_evaluation(args.model_path, args.param_name, args.param_value, args.output_csv)