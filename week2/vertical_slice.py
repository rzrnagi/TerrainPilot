"""
Day 4-5: Vertical slice — camera stream + policy inference in a single loop.

Camera data and policy execution run together: each step the front camera
captures an RGB frame (saved to disk) while the RL policy drives the robot.
This proves the full pipeline shape: camera → save | obs → policy → action → env.

Run:
    cd ~/work/isacc/IsaacLab
    ./isaaclab.sh -p ../TerrainPilot/week2/vertical_slice.py --headless \
        --enable_cameras --onnx_path <path_to_policy.onnx>
"""

import argparse, os, sys
from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../IsaacLab/scripts/reinforcement_learning/rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_path", type=str, required=True, help="Path to exported policy.onnx")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--n_steps",  type=int, default=300)
parser.add_argument("--save_dir", type=str, default="/tmp/terrainpilot_w2_frames")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import cv2
import onnxruntime as ort
import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg_PLAY


@configclass
class Go2WithCameraEnvCfg(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
            update_period=0.1,
            height=240, width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 1e5),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )


def main():
    os.makedirs(args_cli.save_dir, exist_ok=True)

    print(f"\n[TerrainPilot W2] Vertical slice: ONNX policy + camera", flush=True)
    session = ort.InferenceSession(args_cli.onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    cfg = Go2WithCameraEnvCfg()
    cfg.sim.device = args_cli.device or "cuda:0"

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-Play-v0", cfg=cfg)
    obs_dict, _ = env.reset()
    camera = env.unwrapped.scene["camera"]
    start_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].clone()

    saved_frames = 0
    print(f"[TerrainPilot W2] Running {args_cli.n_steps} steps...", flush=True)

    for step in range(args_cli.n_steps):
        # --- policy inference ---
        obs_np = obs_dict["policy"].cpu().numpy().astype(np.float32)
        actions_np = session.run([out_name], {inp_name: obs_np})[0]
        actions = torch.from_numpy(actions_np).to(env.unwrapped.device)

        obs_dict, rew, terminated, truncated, _ = env.step(actions)

        # --- camera capture ---
        rgb = camera.data.output["rgb"]
        if rgb is not None:
            frame = rgb[0, :, :, :3].cpu().numpy()
            cv2.imwrite(os.path.join(args_cli.save_dir, f"frame_{step:04d}.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            saved_frames += 1

        if step % 100 == 0:
            pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
            vel = env.unwrapped.scene["robot"].data.root_lin_vel_b[0]
            disp = (pos - start_pos).norm().item()
            print(f"  step {step:4d} | vel_x={vel[0]:.3f} m/s | "
                  f"reward={rew.item():.3f} | frames_saved={saved_frames} | displacement={disp:.3f} m",
                  flush=True)

        if terminated.any() or truncated.any():
            obs_dict, _ = env.reset()

    end_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    print(f"\n[TerrainPilot W2] Displacement: {(end_pos-start_pos).norm().item():.3f} m", flush=True)
    print(f"[TerrainPilot W2] Frames saved:  {saved_frames} → {args_cli.save_dir}", flush=True)
    print("[TerrainPilot W2] vertical_slice.py complete ✓", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
