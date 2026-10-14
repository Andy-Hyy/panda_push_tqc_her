import gymnasium
import numpy as np
import os
import re
import csv
import multiprocessing as mp
from queue import Empty
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 导入自定义环境包
import panda_mujoco_gym

# --- 配置区 ---
LOGS_DIRECTORY = "./logs"
OUTPUT_CSV_FILE = "batch_evaluation_results_final.csv"
GOAL_RANGES_TO_TEST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_EVAL_EPISODES = 200
EVAL_SEEDS = [0, 42, 123]

def parse_model_info_from_path(path):
    train_type = "unknown"
    train_seed = "unknown"
    dir_name = os.path.basename(os.path.normpath(path))

    seed_match = re.search(r'seed(\d+)', dir_name)
    if seed_match:
        train_seed = int(seed_match.group(1))

    if "dr_spatial" in dir_name:
        train_type = "Spatial_DR"
    elif "noDR" in dir_name and "0.8" in dir_name:
        train_type = "Fixed_0.8"
    else:
        type_match = re.search(r'-(\d\.\d)', dir_name)
        if type_match:
            train_type = f"Fixed_{type_match.group(1)}"
        elif "FrankaPushSparse_v0_TQC" in dir_name and not type_match:
             train_type = "Fixed_0.3"

    return train_type, train_seed

def run_single_test(model_path, train_type, train_seed, eval_seed, test_goal_xy):
    """
    【核心函数】这个函数只执行一个单一的、明确的测试。
    它在一个独立的进程中被调用。
    """
    print(f"    -> 开始测试: train_seed={train_seed}, eval_seed={eval_seed}, test_goal_xy={test_goal_xy}")
    env = None
    try:
        # 1. 为本次测试创建全新的、带有正确参数的环境
        env_id = 'FrankaPushSparse-v0'
        env = gymnasium.make(env_id, goal_xy_range=test_goal_xy)
        
        env.action_space.seed(eval_seed)

        for key in env.observation_space.spaces:
            if isinstance(env.observation_space[key], gymnasium.spaces.Box):
                env.observation_space[key].dtype = np.float32

        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env.seed(eval_seed)

        model = TQC.load(model_path, env=env)
        
        successes = 0
        for i in range(N_EVAL_EPISODES):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
                if done and info[0].get('is_success', False):
                    successes += 1
        
        success_rate = successes / N_EVAL_EPISODES
        print(f"    -->> 完成: train_seed={train_seed}, eval_seed={eval_seed}, test_goal_xy={test_goal_xy}, 成功率: {success_rate:.2%}")

        return {
            'train_type': train_type,
            'train_seed': train_seed,
            'eval_seed': eval_seed,
            'test_goal_xy': test_goal_xy,
            'success_rate': success_rate
        }

    except Exception as e:
        print(f"    -->> 失败: train_seed={train_seed}, eval_seed={eval_seed}, test_goal_xy={test_goal_xy}, 错误: {e}")
        return None
    finally:
        if env is not None:
            env.close()

def evaluation_worker(queue, model_path, train_type, train_seed, eval_seed, test_goal_xy):
    """
    “工人”函数，调用核心测试函数并把结果放入队列。
    """
    result = run_single_test(model_path, train_type, train_seed, eval_seed, test_goal_xy)
    if result:
        queue.put(result)

def main():
    """
    主函数，作为“总指挥”，为每个独立的测试任务创建子进程。
    """
    all_results = []
    
    # 1. 直接指定需要评估的模型文件列表
    tasks = []
    model_files_to_test = [
        "./logs/FrankaPushSparse_v0_TQC_Parallel/noDR_bs512_her4_envs6_seed0_2025-06-26_03-17-28-0.8/final_model.zip",
        "./logs/FrankaPushSparse_v0_TQC_Parallel/noDR_bs512_her4_envs6_seed123_2025-06-26_00-26-38-0.8/final_model.zip",
        "./logs/FrankaPushSparse_v0_TQC_Parallel/noDR_bs512_her4_envs6_seed42_2025-06-25_21-35-39-0.8/final_model.zip"
    ]

    for model_path in model_files_to_test:
        # 确保文件存在
        if not os.path.exists(model_path):
            print(f"警告: 找不到模型文件 {model_path}，跳过此文件。")
            continue

        model_dir = os.path.dirname(model_path)
        train_type, train_seed = parse_model_info_from_path(model_dir)
        
        if train_type != "unknown":
            for eval_seed in EVAL_SEEDS:
                for test_goal_xy in GOAL_RANGES_TO_TEST:
                    tasks.append((model_path, train_type, train_seed, eval_seed, test_goal_xy))

    print(f"--- 共找到 {len(tasks)} 个独立的评估任务 ---")
    
    # 2. 创建一个队列用于进程间通信
    results_queue = mp.Queue()

    # 3. 为每个任务启动一个独立的进程
    for task in tasks:
        model_path, train_type, train_seed, eval_seed, test_goal_xy = task
        print(f"\n=== 准备启动新任务: {os.path.basename(os.path.dirname(model_path))}, eval_seed={eval_seed}, test_goal_xy={test_goal_xy} ===")
        
        process = mp.Process(target=evaluation_worker, args=(results_queue, model_path, train_type, train_seed, eval_seed, test_goal_xy))
        process.start()
        process.join()  # 等待子进程完成

        # 立即从队列中获取结果
        try:
            result_from_run = results_queue.get_nowait()
            all_results.append(result_from_run)
        except Empty:
            pass # 如果子进程出错，队列可能为空

    # 4. 汇总并保存所有结果
    if all_results:
        fieldnames = ['train_type', 'train_seed', 'eval_seed', 'test_goal_xy', 'success_rate']
        # 对结果进行排序，使CSV文件更易读
        all_results.sort(key=lambda x: (x['train_type'], x['train_seed'], x['eval_seed'], x['test_goal_xy']))
        
        with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✅ 所有评测完成！结果已保存至: {OUTPUT_CSV_FILE}")
    else:
        print(f"\n所有评估任务均失败或未返回结果。")

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()