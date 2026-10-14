import os
from panda_mujoco_gym.envs.panda_env import FrankaEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "push.xml")


class FrankaPushEnv(FrankaEnv):
    def __init__(
        self,
        reward_type,
        model_path: str = MODEL_XML_PATH, 
        **kwargs,
    ):
        goal_xy_range = kwargs.pop("goal_xy_range", 0.3)

        super().__init__(
            model_path=model_path, 
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=True,
            distance_threshold=0.05,
            goal_xy_range=goal_xy_range,
            obj_xy_range=0.3,
            goal_x_offset=0.0,
            goal_z_range=0.0,
            **kwargs,
        )