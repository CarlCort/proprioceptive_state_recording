from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking import mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class RecordCommandsCfg:
    """Commands for recording robot state without reference motion."""

    robot_state = mdp.RobotStateCommandCfg(
        asset_name="robot",
        reference_body="torso_link",
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    )


@configclass
class EmptyCfg:
    """Empty configuration container used for rewards and terminations."""

    pass


@configclass
class G1RecordEnvCfg(TrackingEnvCfg):
    """Environment configuration for recording G1 robot state."""

    commands: RecordCommandsCfg = RecordCommandsCfg()
    rewards: EmptyCfg = EmptyCfg()
    terminations: EmptyCfg = EmptyCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
