# stack.py

import os
import numpy as np
# 导入父类 FrankaEnv
from panda_mujoco_gym.envs.panda_env import FrankaEnv 

# 1. 定义新的XML文件路径
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "stack.xml")

# 2. 创建新类，继承自FrankaEnv
class FrankaStackEnv(FrankaEnv):
    def __init__(self, reward_type="sparse", **kwargs):
        # 3. 初始化父类，并传入Stack任务的特定配置
        super().__init__(
            model_path=MODEL_XML_PATH,
            reward_type=reward_type,
            n_substeps=25,
            block_gripper=False,  # 抓取任务，不锁夹爪
            # 注意：这里我们不再需要n_actions=4，因为父类会根据block_gripper自动判断
            distance_threshold=0.05,
            **kwargs,
        )

    # 4. 重写(Override)真正有差异的函数

    def _get_obs(self) -> dict:
        # Stack任务需要观测两个物体，这与父类的逻辑不同，必须重写
        obs = super()._get_obs() # 先获取父类处理好的机器人观测
        
        # 获取obj1和obj2的位置信息
        obj1_pos = self.sim.data.get_body_xpos("obj").copy()
        obj2_pos = self.sim.data.get_body_xpos("obj2").copy()
        
        # 将两个物体的状态信息加入到观测中
        obs["observation"] = np.concatenate([obs["observation"], obj1_pos, obj2_pos])
        
        # "achieved_goal" 是我们想要控制的物体（obj2）的当前位置
        obs["achieved_goal"] = obj2_pos
        
        return obs

    def _reset_sim(self) -> bool:
        # Stack任务需要重置两个物体，父类只会重置一个，必须重写
        did_reset_sim = super()._reset_sim()
        if not did_reset_sim:
            return False

        # 随机放置第一个物体 (底座)
        obj1_pos = self._sample_object_pos()
        self.sim.data.set_joint_qpos("obj_joint", np.concatenate([obj1_pos, [1.0, 0.0, 0.0, 0.0]]))

        # 随机放置第二个物体 (要被抓取的)
        while True:
            obj2_pos = self._sample_object_pos()
            if np.linalg.norm(obj1_pos - obj2_pos) > 0.1: # 保证初始有一定距离
                break
        self.sim.data.set_joint_qpos("obj2_joint", np.concatenate([obj2_pos, [1.0, 0.0, 0.0, 0.0]]))
        
        self.sim.forward()
        return True

    def _sample_goal(self) -> np.ndarray:
        # Stack任务的目标是第一个物体上方，父类的随机采样逻辑不适用，必须重写
        # 这个函数在环境重置时被调用，用来生成本回合的目标
        obj1_pos = self.sim.data.get_body_xpos("obj").copy()
        
        # 假设物体是 0.02 * 2 = 0.04 高
        object_height = 0.04 
        goal = obj1_pos + np.array([0.0, 0.0, object_height])
        return goal