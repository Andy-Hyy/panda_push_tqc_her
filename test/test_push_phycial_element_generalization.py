import subprocess
import os
import re
import argparse # 使用argparse来处理命令行参数

# --- 物理参数测试的定义 ---
# 定义了所有单因素物理测试的参数名和对应的测试值列表
PHYSICS_PARAMS = {
    "object_mass": [0.05, 0.1, 0.5, 1.0, 2.0, 3.0],
    "object_friction": [0.05, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    "joint_damping_scale": [0.001, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3],
    "actuator_gain_scale": [0, 1, 2, 3, 5, 6, 7, 10]
}
PHYSICS_WORKER_SCRIPT = "worker_single_physics.py" # 单因素测试使用的 "工人" 脚本

# --- 多参数联合扰动测试的定义 ---
MULTI_PARAM_WORKER_SCRIPT = "worker_multi_random.py" # 多因素测试使用的 "工人" 脚本


def find_models(root_dir):
    """查找所有final_model.zip或std_model.zip文件"""
    models = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f == "final_model.zip" or f == "std_model.zip":
                models.append(os.path.join(dirpath, f))
    return models

def run_physics_tests(models_to_test):
    """(功能已补全) 运行单一物理参数泛化测试"""
    for model_path in models_to_test:
        seed_str = "unknown_seed"
        match = re.search(r'seed(\d+)', model_path)
        if match:
            seed_str = f"seed{match.group(1)}"

        # 遍历总清单中的每一个物理参数
        for param_name, param_values in PHYSICS_PARAMS.items():
            output_csv = f"eval_phy_{param_name}_{seed_str}.csv"
            print(f"\n--- 开始(物理)测试 '{param_name}'，模型 '{seed_str}' ---")
            print(f"--- 结果将保存至: {output_csv} ---")
            
            # 创建并写入CSV表头，确保每次都是新文件
            with open(output_csv, "w") as f:
                f.write("param_name,value,success_rate\n")

            # 遍历该参数下的每一个具体数值
            for value in param_values:
                print(f"  -> 正在测试 {param_name} = {value}")
                command = [
                    "python",
                    PHYSICS_WORKER_SCRIPT,
                    "--model-path", model_path,
                    "--param-name", param_name,
                    "--param-value", str(value), # 确保值为字符串
                    "--output-csv", output_csv,
                ]
                try:
                    # check=True 表示如果子进程返回非零退出码（即出错），则会抛出异常
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"    [!! 错误 !!] 子进程运行失败: {e}")
                except FileNotFoundError:
                    print(f"    [!! 错误 !!] 找不到 'python' 或 '{PHYSICS_WORKER_SCRIPT}'。")
                    # 如果找不到worker，就没必要继续当前参数的循环了
                    break

def run_multi_param_tests(models_to_test):
    """运行多参数联合扰动鲁棒性测试"""
    for model_path in models_to_test:
        seed_str = "unknown_seed"
        match = re.search(r'seed(\d+)', model_path)
        if match:
            seed_str = f"seed{match.group(1)}"
        elif "std_model" in model_path:
            seed_str = "std"

        output_csv = f"eval_multi_param_{seed_str}.csv"
        print(f"\n--- 开始(多参数)测试模型 '{model_path}'，结果保存至 {output_csv} ---")

        command = [
            "python",
            MULTI_PARAM_WORKER_SCRIPT,
            "--model-path", model_path,
            "--output-csv", output_csv,
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"    [!! 错误 !!] 子进程运行失败: {e}")
        except FileNotFoundError:
            print(f"    [!! 错误 !!] 找不到 'python' 或 '{MULTI_PARAM_WORKER_SCRIPT}'。")
            return

def main():
    parser = argparse.ArgumentParser(description="模型泛化能力与鲁棒性测试启动器")
    parser.add_argument("--models-dir", type=str, default="./logs", help="存放模型的根目录")
    parser.add_argument("--test-type", type=str, required=True, choices=['physics', 'multi'], help="测试类型: 'physics' (单物理参数) 或 'multi' (多参数联合扰动)")
    args = parser.parse_args()
    
    models_to_test = find_models(args.models_dir)
    if not models_to_test:
        print(f"在 {args.models_dir} 中没有找到任何模型文件。")
        return
    print(f"找到 {len(models_to_test)} 个模型将要进行评测。")

    # (功能已补全) 根据命令行参数选择执行相应的测试函数
    if args.test_type == 'physics':
        run_physics_tests(models_to_test)
    elif args.test_type == 'multi':
        run_multi_param_tests(models_to_test)

if __name__ == "__main__":
    main()