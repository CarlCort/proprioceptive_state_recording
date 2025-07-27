"""Record proprioceptive robot state without running RL."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# command line arguments
parser = argparse.ArgumentParser(description="Record proprioceptive state from an Isaac Lab environment")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Gym environment name")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to record")
parser.add_argument("--out_file", type=str, default="recordings/proprioceptive.npz", help="Path to output npz file")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")

# append AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()

sys.argv = [sys.argv[0]] + unknown

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib
import numpy as np
import torch

# Import tasks so that environments are registered
import whole_body_tracking.tasks  # noqa: F401

# resolve env_cfg entry point from gym registration
spec = gym.spec(args_cli.task)
entry = spec.kwargs.get("env_cfg_entry_point")
if isinstance(entry, str):
    mod_name, cls_name = entry.split(":")
    env_cfg_cls = getattr(importlib.import_module(mod_name), cls_name)
else:
    env_cfg_cls = entry
env_cfg = env_cfg_cls()
env_cfg.scene.num_envs = args_cli.num_envs

# create environment
env = gym.make(args_cli.task, cfg=env_cfg)

# access robot
robot = env.unwrapped.scene["robot"]

# logging containers
log = {"joint_pos": [], "joint_vel": [], "body_pos_w": [], "body_quat_w": [], "body_lin_vel_w": [], "body_ang_vel_w": []}

obs, _ = env.reset()
zero_action = torch.zeros_like(env.action_manager.action_tensor)

for _ in range(args_cli.num_steps):
    obs, _, _, _ = env.step(zero_action)
    log["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
    log["joint_vel"].append(robot.data.joint_vel.cpu().numpy())
    log["body_pos_w"].append(robot.data.body_pos_w.cpu().numpy())
    log["body_quat_w"].append(robot.data.body_quat_w.cpu().numpy())
    log["body_lin_vel_w"].append(robot.data.body_lin_vel_w.cpu().numpy())
    log["body_ang_vel_w"].append(robot.data.body_ang_vel_w.cpu().numpy())

for k in log:
    log[k] = np.stack(log[k], axis=0)

os.makedirs(os.path.dirname(args_cli.out_file), exist_ok=True)
np.savez(args_cli.out_file, **log)

env.close()
simulation_app.close()
