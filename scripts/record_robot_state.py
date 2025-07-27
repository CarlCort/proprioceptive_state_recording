"""Record proprioceptive robot state without running RL."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record robot state from an environment.")
parser.add_argument(
    "--task",
    type=str,
    default="Recording-Flat-G1-v0",
    help="Name of the gym environment",
)
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to record")
parser.add_argument(
    "--output", type=str, default="recording.npz", help="File to save recorded data"
)

# add app launcher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest of the imports after simulation app is created."""

import gymnasium as gym
import numpy as np
import torch

import whole_body_tracking.tasks  # noqa: F401


def main() -> None:
    env = gym.make(args.task)
    robot = env.scene["robot"]

    log = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "fps": [int(1.0 / env.physics_dt)],
    }

    obs, _ = env.reset()
    zero_action = torch.zeros_like(env.action_manager.get_term("joint_pos")._scale)

    for _ in range(args.steps):
        obs, _, terminated, _ = env.step(zero_action)

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        if terminated.any():
            env.reset()

    np.savez(
        args.output,
        **{k: np.stack(v) if k != "fps" else np.array(v) for k, v in log.items()}
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
