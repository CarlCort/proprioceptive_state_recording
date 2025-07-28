import numpy as np
import torch

from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

@configclass
class RobotStateSensorCfg(SensorBaseCfg):
    asset_name: str = MISSING # Robot asset name in the simulation
    reference_body: str = MISSING # Reference body link name within robot for body-frame measurements
    body_names: list[str] = MISSING # names of links to track

class RobotStateSensor(SensorBase):
    cfg: RobotStateSensorCfg

    def __init__(self, cfg: RobotStateSensorCfg, sim):
        super().__init__(cfg, sim)
        self.robot = sim.scene[cfg.asset_name]
        # …initialize body_indexes, ref_index just like your old CommandTerm…:contentReference[oaicite:3]{index=3}
        self.robot_ref_body_index = self.robot.body_names.index(self.cfg.reference_body)
        self.body_indexes = torch.tensor(self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
                                         dtype=torch.long, device=self.device)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_body_frame_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_ref_body_index]

    @property
    def robot_body_frame_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_ref_body_index]

    @property
    def robot_body_frame_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_ref_body_index]

    @property
    def robot_body_frame_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_ref_body_index]

    def read(self) -> dict[str, torch.Tensor]:
        """Called automatically at cfg.update_period intervals."""
        return {
            'joint_positions': self.robot_joint_pos,
            'joint_velocities': self.robot_joint_vel,
            'link_positions_w': self.robot_body_pos_w,
            'link_quaternions_w': self.robot_body_quat_w,
            'link_linear_velocities_w': self.robot_body_lin_vel_w,
            'link_angular_velocities_w': self.robot_body_ang_vel_w,
            'body_ref_frame_position_w': self.robot_body_frame_pos_w,
            'body_ref_frame_quaternion_w': self.robot_body_frame_quat_w,
            'body_ref_frame_linear_velocity_w': self.robot_body_frame_lin_vel_w,
            'body_ref_frame_angular_velocity_w': self.robot_body_frame_ang_vel_w
        }
