"""
Day 1-2: Velocity command → Go2 env → verify state changes.

Spawns 1 Go2 on flat terrain, forces the velocity command to 0.5 m/s
forward, and runs with zero joint-position actions for 300 steps.
No policy needed — this is a plumbing test only.

Run:
    cd ~/work/isacc/IsaacLab
    ./isaaclab.sh -p ../TerrainPilot/week1/velocity_cmd.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401 — registers all tasks
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg_PLAY,
)

VX_MPS   = 0.5   # forward velocity command (m/s)
VY_MPS   = 0.0
YAW_RPS  = 0.0
N_STEPS  = 300
LOG_EVERY = 50


def main():
    cfg: ManagerBasedRLEnvCfg = UnitreeGo2FlatEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-Play-v0", cfg=cfg)
    obs, _ = env.reset()

    # default joint-position action (12-DOF Go2): no locomotion, just standing
    action = torch.zeros(1, env.action_space.shape[-1], device=env.unwrapped.device)

    print(f"\n[TerrainPilot W1] Commanding vx={VX_MPS} m/s for {N_STEPS} steps\n")

    for step in range(N_STEPS):
        # override the velocity command tensor each step
        env.unwrapped.command_manager.get_term("base_velocity").vel_command_b[:] = torch.tensor(
            [[VX_MPS, VY_MPS, YAW_RPS]], device=env.unwrapped.device
        )

        obs, rew, terminated, truncated, info = env.step(action)

        if step % LOG_EVERY == 0:
            robot = env.unwrapped.scene["robot"]
            base_pos  = robot.data.root_pos_w[0]          # (3,)  world pos
            base_vel  = robot.data.root_lin_vel_b[0]       # (3,)  body-frame vel
            joint_pos = robot.data.joint_pos[0]            # (12,)
            print(
                f"  step {step:4d} | "
                f"pos=[{base_pos[0]:.2f},{base_pos[1]:.2f},{base_pos[2]:.2f}] m | "
                f"vel_x={base_vel[0]:.3f} m/s | "
                f"reward={rew.item():.3f}",
                flush=True,
            )

        if terminated.any() or truncated.any():
            obs, _ = env.reset()

    env.close()
    print("\n[TerrainPilot W1] velocity_cmd.py complete ✓")


if __name__ == "__main__":
    main()
    simulation_app.close()
