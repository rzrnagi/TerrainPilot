"""
Day 3: Camera stream → RGB images → save 100 frames to disk.

Attaches a front camera to the Go2's base, runs the env for 100 steps,
saves each RGB frame as a PNG and prints timestamps.

Run:
    cd ~/work/isacc/IsaacLab
    ./isaaclab.sh -p ../TerrainPilot/week1/camera_stream.py --headless --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="/tmp/terrainpilot_frames")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time
import torch
import numpy as np
import cv2
import gymnasium as gym

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg_PLAY,
)

N_FRAMES = 100


@configclass
class Go2WithCameraEnvCfg(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
            update_period=0.1,   # 10 Hz
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, 0.0, 0.05),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )


def main():
    os.makedirs(args_cli.save_dir, exist_ok=True)

    cfg = Go2WithCameraEnvCfg()
    cfg.sim.device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-Play-v0", cfg=cfg)
    obs, _ = env.reset()

    action = torch.zeros(1, env.action_space.shape[-1], device=env.unwrapped.device)
    camera = env.unwrapped.scene["camera"]

    print(f"\n[TerrainPilot W1] Saving {N_FRAMES} frames to {args_cli.save_dir}\n")

    saved = 0
    step = 0
    timestamps = []

    while saved < N_FRAMES:
        env.step(action)
        step += 1

        rgb = camera.data.output["rgb"]   # (N_envs, H, W, 4) RGBA uint8
        if rgb is None:
            continue

        frame = rgb[0, :, :, :3].cpu().numpy()   # (H, W, 3) RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ts = time.time()
        timestamps.append(ts)
        path = os.path.join(args_cli.save_dir, f"frame_{saved:04d}.png")
        cv2.imwrite(path, frame_bgr)
        saved += 1

        if saved % 10 == 0:
            print(f"  saved {saved}/{N_FRAMES}  shape={frame.shape}  ts={ts:.3f}")

    env.close()

    # verify timestamps are sane (~10 Hz)
    if len(timestamps) > 1:
        diffs = np.diff(timestamps)
        print(f"\n[TerrainPilot W1] Frame interval: mean={diffs.mean()*1000:.1f} ms  "
              f"min={diffs.min()*1000:.1f} ms  max={diffs.max()*1000:.1f} ms")

    print(f"\n[TerrainPilot W1] camera_stream.py complete ✓  ({N_FRAMES} frames in {args_cli.save_dir})")


if __name__ == "__main__":
    main()
    simulation_app.close()
