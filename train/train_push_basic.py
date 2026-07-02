import os
import sys
import time
import argparse  # 导入用于处理命令行参数的库
import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.her import HerReplayBuffer

# 确保能找到自定义环境库
# sys.path.append('...')
import panda_mujoco_gym 

# 导入 torch 库，用于 GPU 检测
import torch 

# 确保能找到自定义环境库
# sys.path.append('...')
import panda_mujoco_gym 



def train(seed, batch_size, n_sampled_goal):
    """
    一个包含完整训练逻辑的函数，接收种子和超参数作为输入。
    """
    print(f"--- 开始新一轮训练 ---")
    print(f"随机种子 (Seed): {seed}")
    print(f"批量大小 (Batch Size): {batch_size}")
    print(f"HER采样数 (n_sampled_goal): {n_sampled_goal}")

    # --- 1. 创建环境 ---
    env = gym.make("FrankaPushSparse-v0")

    # --- 2. 设置独立的日志和模型路径 ---
    # 为每次训练（特别是不同种子）创建独立的文件夹
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    config_name = f"batch{batch_size}_her{n_sampled_goal}"
    log_dir = f"logs/FrankaPushSparse_v0_TQC/{config_name}_seed{seed}_{run_timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    
    final_model_path = os.path.join(log_dir, "final_model.zip")

    # --- 3. 创建回调 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(log_dir, "checkpoints/"),
        name_prefix="tqc_frankapush",
    )

    # --- 4. 创建TQC模型 ---
    model = TQC(
        policy="MultiInputPolicy", # 指定策略网络类型。 "MultiInputPolicy" 适用于输入是字典格式的环境 (例如，包含 'observation', 'desired_goal', 和 'achieved_goal')，常用于机器人任务。
        env=env, # 传入已经实例化的强化学习环境，智能体将在这个环境中进行交互和学习
        seed=seed,  # 设置随机种子。以确保实验结果的可复现性。固定的种子可以保证每次运行时，网络的初始权重和数据采样等随机过程都是相同的。
        learning_rate=1e-3, # 设置优化器（通常是Adam）的学习率。它控制着在每次梯度下降时更新网络权重的步长。
        buffer_size=1000000,  # 设置经验回放缓冲区（Replay Buffer）的大小。它用于存储智能体过去的经验（状态、动作、奖励等），TQC作为一种离策略（off-policy）算法，会从中采样数据进行训练。
        batch_size=batch_size, # 使用传入的参数， # 每一轮训练时，从经验回放缓冲区中采样的样本数量
        tau=0.05,   # 目标网络（Target Network）的软更新系数。为了稳定训练，TQC会使用一个更新缓慢的目标网络
        gamma=0.95, # 折扣因子（Discount Factor）。它决定了未来奖励的重要性，用于平衡即时奖励和长期奖励。值越接近1，智能体越看重未来的奖励。
        top_quantiles_to_drop_per_net=2,  # 这是TQC算法的核心参数之一。TQC会训练多个评论家（Critic）网络，每个网络预测一个回报分布（分位数）。为了抑制Q值的过高估计，它会从每个网络中丢弃掉最高的几个分位数，这里设置为2。
        ent_coef="auto",  # 熵系数（Entropy Coefficient）。用于在最大化奖励的同时，鼓励策略网络探索更多可能性（即保持较高的熵）。设置为 "auto" 意味着算法会自动调整这个系数，使其达到一个目标熵，这通常比手动设置固定值更稳定。
        replay_buffer_class=HerReplayBuffer,  # 指定经验回放缓冲区的类别。这里使用了 Hindsight Experience Replay (HER)，这是一种专门用于解决稀疏奖励问题的技术，通过事后“假设”不同的目标来创造成功的经验，极大地提高了学习效率。
        replay_buffer_kwargs={
            "n_sampled_goal": n_sampled_goal, # 使用传入的参数， # 对于缓冲区中的每一条经验，HER会额外采样的“伪”目标数量。
            "goal_selection_strategy": "future", # 设定如何选择伪目标。"future" 策略表示从当前经验在同一回合（episode）中未来的状态里选择目标，这是最常用且有效的策略。
        },
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]), # 定义神经网络的结构。pi (policy) 代表策略网络（Actor），qf (Q-function) 代表价值网络（Critic）。这里两者都设置为包含三个隐藏层，每层有256个神经元。
            n_quantiles=25 # TQC中每个评论家（Critic）网络要预测的分位数数量。评论家不只预测一个单一的Q值，而是预测一个由25个值组成的回报分布。
        ),
        tensorboard_log=log_dir,  # 指定TensorBoard日志的保存路径。训练过程中的奖励、损失等数据会被记录下来，方便后续进行可视化分析和调试。
        verbose=1,  # 设置日志的详细程度。1 表示会在控制台打印训练进度等信息；0 表示静默模式。
    )

    # --- 5. 开始学习 ---
    print(f"日志将保存在: {log_dir}")
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save(final_model_path)
    print(f"\n训练完成！模型已保存至: {final_model_path}")


if __name__ == "__main__":
    # --- 6. 使用 argparse 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="为FrankaPush任务训练TQC模型")
    # 添加 --seed 参数，类型为整数，默认为0
    parser.add_argument("--seed", type=int, default=0, help="训练使用的随机种子")
    # 添加 --batch_size 参数
    parser.add_argument("--batch_size", type=int, default=512, help="批量大小")
    # 添加 --n_sampled_goal 参数
    parser.add_argument("--n_sampled_goal", type=int, default=4, help="HER采样数")

    args = parser.parse_args()

    # 调用主训练函数，并传入解析到的参数
    train(seed=args.seed, batch_size=args.batch_size, n_sampled_goal=args.n_sampled_goal)
