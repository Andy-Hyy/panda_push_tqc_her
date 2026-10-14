import os
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
import numpy as np
from rich.console import Console
from rich.table import Table
import panda_mujoco_gym
import subprocess
import sys

# ==============================================================================
# 1. 配置参数
# ==============================================================================
SEEDS = [0, 42, 123]
TEST_PARAMS = {
    "object_mass": [0.05, 0.1, 0.5, 1.0, 2.0, 3.0],
    "object_friction": [0.05, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    "joint_damping_scale": [0.001, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3],
    "actuator_gain_scale": [0, 1, 2, 3, 5, 6, 7, 10]
}

# ==============================================================================
# 2. 核心功能: XML修改与安全备份
# ==============================================================================
def get_xml_paths():
    """自动查找 panda_mujoco_gym 包内的 XML 文件路径"""
    try:
        gym_path = os.path.dirname(panda_mujoco_gym.__file__)
        push_xml_path = os.path.join(gym_path, "assets", "push.xml")
        mocap_xml_path = os.path.join(gym_path, "assets", "panda_mocap.xml")
        if not os.path.exists(push_xml_path) or not os.path.exists(mocap_xml_path):
            raise FileNotFoundError
        return push_xml_path, mocap_xml_path
    except (ImportError, FileNotFoundError):
        print("错误: 无法自动定位 'push.xml' 或 'panda_mocap.xml'。")
        exit()

@contextmanager
def modify_xml_safely(xml_path):
    """安全地修改XML文件，并在操作后恢复原文件。"""
    original_content = ""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        tree = ET.parse(xml_path)
        yield tree.getroot()
        tree.write(xml_path, encoding='utf-8', xml_declaration=False)
    finally:
        if original_content:
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

# ==============================================================================
# 4. 主测试流程
# ==============================================================================
def main():
    """执行完整的物理泛化测试流程"""
    console = Console()
    worker_script_path = os.path.join(os.path.dirname(__file__), "worker.py")

    if not os.path.exists(worker_script_path):
        console.print(f"[bold red]错误: 未找到工作者脚本 'worker.py'。请确保它与本脚本在同一目录下。[/bold red]")
        return
        
    PUSH_XML_PATH, MOCAP_XML_PATH = get_xml_paths()
    results = {}
    start_time = time.time()
    
    test_suite = {
        "object_mass": (PUSH_XML_PATH, ".//geom[@name='obj_geom']", "mass", str),
        "object_friction": (PUSH_XML_PATH, ".//geom[@name='obj_geom']", "friction", lambda v: f'{v} {v*0.1} {v*0.005}'),
        "joint_damping_scale": (MOCAP_XML_PATH, ".//joint", "damping", float),
        "actuator_gain_scale": (MOCAP_XML_PATH, ".//actuator/general", "gainprm", float)
    }

    for param_name, (xml_path, element_selector, attr_name, val_formatter) in test_suite.items():
        console.print(f"\n--- [bold green]开始测试组: {param_name}[/bold green] ---")
        results[param_name] = {}
        
        defaults = {}
        if "_scale" in param_name:
            with modify_xml_safely(xml_path) as root:
                for element in root.findall(element_selector):
                    name = element.get('name')
                    if name and element.get(attr_name) is not None:
                        defaults[name] = element.get(attr_name)

        for value in TEST_PARAMS[param_name]:
            console.print(f"  设置 {param_name} = {value}")
            
            with modify_xml_safely(xml_path) as root:
                elements = root.findall(element_selector)
                for element in elements:
                    if "_scale" in param_name:
                        name = element.get('name')
                        if name in defaults:
                            base_values = [float(p) for p in defaults[name].split()]
                            scaled_values = [str(p * value) for p in base_values]
                            element.set(attr_name, " ".join(scaled_values))
                    else:
                        element.set(attr_name, val_formatter(value))

            try:
                python_executable = sys.executable
                process = subprocess.run(
                    [python_executable, worker_script_path],
                    capture_output=True, text=True, check=True, timeout=600
                )

                # --- 关键修复：增加解析的鲁棒性 ---
                output_lines = process.stdout.strip().split('\n')
                seed_results = []
                for line in output_lines:
                    try:
                        # 尝试将行转换为浮点数
                        rate = float(line)
                        seed_results.append(rate)
                    except ValueError:
                        # 如果转换失败（例如，该行是 "Wrapping..."），则忽略它
                        continue
                # --- 修复结束 ---

                if len(seed_results) != len(SEEDS):
                    raise ValueError(f"工作者脚本返回了 {len(seed_results)} 个有效数字，但期望得到 {len(SEEDS)} 个。")

                results[param_name][value] = seed_results
                for seed, rate in zip(SEEDS, seed_results):
                     console.print(f"    - Seed {seed}: 成功率 = {rate:.2f}%")

            except subprocess.TimeoutExpired:
                 console.print(f"[bold red]  !-> 测试值 {value} 超时! [/bold red]")
                 results[param_name][value] = [-1.0] * len(SEEDS)
            except Exception as e:
                # 检查 e 是否有 stderr 属性，以防是其他类型的错误
                stderr_output = e.stderr if hasattr(e, 'stderr') else 'N/A'
                console.print(f"[bold red]  !-> 测试值 {value} 失败. 错误: {e}[/bold red]")
                console.print(f"[bold yellow]工作者脚本的错误输出:\n{stderr_output}[/bold yellow]")
                results[param_name][value] = [0.0] * len(SEEDS)
    
    total_time = time.time() - start_time
    console.print(f"\n[bold]全部测试完成，总耗时: {total_time:.2f} 秒。[/bold]")

    # --- 打印最终结果报告 ---
    console.print("\n" + "="*50)
    console.print("[bold cyan]物理泛化测试最终报告[/bold cyan]")
    console.print("="*50 + "\n")

    for param_name, param_results in results.items():
        table = Table(title=f"泛化测试: {param_name}")
        table.add_column("参数值", justify="right", style="cyan", no_wrap=True)
        for seed in SEEDS:
            table.add_column(f"Seed {seed} (%)", justify="center")
        table.add_column("平均成功率 (%)", justify="right", style="bold green")
        for value, seed_rates in param_results.items():
            valid_rates = [r for r in seed_rates if r >= 0]
            avg_rate_str = f"{np.mean(valid_rates):.2f}" if valid_rates else "N/A"
            row_data = [f"{value:.3f}"] + [f"{r:.2f}" for r in seed_rates] + [avg_rate_str]
            table.add_row(*row_data)
        console.print(table)

if __name__ == "__main__":
    main()