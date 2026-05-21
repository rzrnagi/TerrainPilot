"""
Day 5-6: Open-loop controller → log run → replay and verify offline.

Command schedule: forward 0.5 m/s for 3 s → turn left 0.5 rad/s for 2 s → repeat.
Logs full telemetry via logger.py and prints a replay summary at the end.

Run:
    cd ~/work/isacc/IsaacLab
    ./isaaclab.sh -p ../TerrainPilot/week1/open_loop_run.py --headless

    # To just replay an existing log without re-running the sim:
    python ../TerrainPilot/week1/open_loop_run.py --replay_only --log_path /tmp/terrainpilot_w1.npz
"""

import argparse
import sys
import os

# Allow --replay_only without Isaac Sim when just replaying logs
_replay_only = "--replay_only" in sys.argv

if not _replay_only:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="/tmp/terrainpilot_w1.npz")
    parser.add_argument("--cycles",   type=int, default=3, help="forward+turn cycles to run")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_only", action="store_true")
    parser.add_argument("--log_path", type=str, default="/tmp/terrainpilot_w1.npz")
    args_cli = parser.parse_args()

# Add week1 dir to path so logger.py is importable
sys.path.insert(0, os.path.dirname(__file__))
from logger import TelemetryLogger

if _replay_only:
    TelemetryLogger.replay_stats(args_cli.log_path)
    sys.exit(0)

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg_PLAY,
)

# Control frequency is 50 Hz (decimation=4, physics dt=0.005 → env dt=0.02 s)
ENV_DT_S      = 0.02
FORWARD_S     = 3.0
TURN_S        = 2.0
VX_FORWARD    = 0.5    # m/s
YAW_TURN      = 0.5    # rad/s (positive = left)

FORWARD_STEPS = int(FORWARD_S / ENV_DT_S)
TURN_STEPS    = int(TURN_S    / ENV_DT_S)
CYCLE_STEPS   = FORWARD_STEPS + TURN_STEPS


def build_command_schedule(n_cycles: int, device: str):
    """Returns (N, 3) tensor of [vx, vy, yaw] commands for n_cycles."""
    fwd  = torch.tensor([[VX_FORWARD, 0.0, 0.0]], device=device).repeat(FORWARD_STEPS, 1)
    turn = torch.tensor([[0.0, 0.0, YAW_TURN]],  device=device).repeat(TURN_STEPS,   1)
    cycle = torch.cat([fwd, turn], dim=0)
    return cycle.repeat(n_cycles, 1)   # (N_total, 3)


def main():
    cfg = UnitreeGo2FlatEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-Play-v0", cfg=cfg)
    obs, _ = env.reset()

    device = env.unwrapped.device
    schedule = build_command_schedule(args_cli.cycles, device)
    n_steps  = schedule.shape[0]

    action  = torch.zeros(1, env.action_space.shape[-1], device=device)
    logger  = TelemetryLogger(args_cli.log_path)
    robot   = env.unwrapped.scene["robot"]

    print(f"\n[TerrainPilot W1] Open-loop run: {args_cli.cycles} cycles × "
          f"({FORWARD_S}s fwd + {TURN_S}s turn) = {n_steps} steps\n")

    for step in range(n_steps):
        cmd = schedule[step].unsqueeze(0)   # (1, 3)

        env.unwrapped.command_manager.get_term("base_velocity").vel_command_b[:] = cmd

        obs, rew, terminated, truncated, _ = env.step(action)
        logger.record(robot.data, cmd)

        phase = "FWD " if step % CYCLE_STEPS < FORWARD_STEPS else "TURN"
        if step % 50 == 0:
            pos = robot.data.root_pos_w[0]
            print(f"  [{phase}] step {step:4d} | "
                  f"pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] | "
                  f"cmd=[{cmd[0,0]:.1f},{cmd[0,2]:.1f}] | "
                  f"rew={rew.item():.3f}")

        if terminated.any() or truncated.any():
            obs, _ = env.reset()

    log_path = logger.save()
    env.close()

    print("\n--- Replay verification ---")
    TelemetryLogger.replay_stats(log_path)
    print("\n[TerrainPilot W1] open_loop_run.py complete ✓")


if __name__ == "__main__":
    main()
    if not _replay_only:
        simulation_app.close()
