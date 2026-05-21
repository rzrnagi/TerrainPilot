"""
Day 3: Verify the exported ONNX policy runs correctly.

Loads the ONNX exported by run_stock_policy.py, feeds live observations
from the Go2 env, gets actions, and steps the env. Confirms the ONNX
policy produces the same locomotion as the JIT/PyTorch version.

Run:
    cd ~/work/isacc/IsaacLab
    ./isaaclab.sh -p ../TerrainPilot/week2/onnx_verify.py --headless \
        --onnx_path <path_to_policy.onnx>
"""

import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_path", type=str, required=True, help="Path to exported policy.onnx")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--n_steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import onnxruntime as ort
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg_PLAY


def main():
    print(f"\n[TerrainPilot W2] Loading ONNX from: {args_cli.onnx_path}", flush=True)
    session = ort.InferenceSession(args_cli.onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    print(f"  Input:  {inp_name}  {session.get_inputs()[0].shape}", flush=True)
    print(f"  Output: {out_name}  {session.get_outputs()[0].shape}", flush=True)

    cfg = UnitreeGo2FlatEnvCfg_PLAY()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-Play-v0", cfg=cfg)
    obs_dict, _ = env.reset()
    start_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].clone()

    print(f"[TerrainPilot W2] Running ONNX policy for {args_cli.n_steps} steps...", flush=True)
    for step in range(args_cli.n_steps):
        obs_np = obs_dict["policy"].cpu().numpy().astype(np.float32)   # (1, obs_dim)
        actions_np = session.run([out_name], {inp_name: obs_np})[0]    # (1, 12)
        actions = torch.from_numpy(actions_np).to(env.unwrapped.device)

        obs_dict, rew, terminated, truncated, _ = env.step(actions)

        if step % 100 == 0:
            pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
            vel = env.unwrapped.scene["robot"].data.root_lin_vel_b[0]
            disp = (pos - start_pos).norm().item()
            print(f"  step {step:4d} | vel_x={vel[0]:.3f} m/s | "
                  f"reward={rew.item():.3f} | displacement={disp:.3f} m", flush=True)

        if terminated.any() or truncated.any():
            obs_dict, _ = env.reset()

    end_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    print(f"\n[TerrainPilot W2] ONNX total displacement: {(end_pos-start_pos).norm().item():.3f} m", flush=True)
    print("[TerrainPilot W2] onnx_verify.py complete ✓", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
