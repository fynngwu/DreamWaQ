import torch
from isaacgym import gymtorch
from legged_gym.envs.base.legged_robot import LeggedRobot

class DogV2Robot(LeggedRobot):
    def _reward_hip_default(self):
        hip_err = torch.sum((self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]) ** 2, dim=1)
        return hip_err
