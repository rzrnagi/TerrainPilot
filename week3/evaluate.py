"""
Day 7: Compare stock vs fine-tuned policy on sloped terrain with payload.

Runs both policies for N_STEPS steps in the slope+payload env and reports:
  - Success rate (didn't fall: base height > threshold for full episode)
  - Mean velocity tracking error vs commanded velocity
  - Mean displacement

Uses JIT (.pt) policies for multi-env evaluation (ONNX was exported batch=1).

Run:
    cd ~/work/isacc/IsaacLab
    ./isaaclab.sh -p ../TerrainPilot/week3/evaluate.py --headless \
        --stock_jit      logs/rsl_rl/unitree_go2_flat/.../exported/policy.pt \
        --finetuned_jit  logs/rsl_rl/go2_slope_payload/.../exported/policy.pt
"""

import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--stock_jit",     type=str, required=True, help="JIT .pt from Week 2 (flat policy)")
parser.add_argument("--finetuned_jit", type=str, required=True, help="JIT .pt from Week 3 fine-tuning")
parser.add_argument("--n_steps",  type=int, default=500)
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(__file__))
from terrain_payload_env_cfg import UnitreeGo2SlopePayloadEnvCfg_EVAL  # noqa: E402

BASE_HEIGHT_THRESHOLD = 0.12


def run_eval(label: str, policy, cfg, n_steps: int, device: str):
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    obs_dict, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    n_envs = cfg.scene.num_envs

    start_pos = robot.data.root_pos_w.clone()
    falls = torch.zeros(n_envs, dtype=torch.bool, device=device)
    vel_errors = []

    for step in range(n_steps):
        with torch.inference_mode():
            actions = policy(obs_dict["policy"])

        obs_dict, _, terminated, truncated, _ = env.step(actions)

        base_h = robot.data.root_pos_w[:, 2]
        falls |= base_h < BASE_HEIGHT_THRESHOLD

        cmd_vx = env.unwrapped.command_manager.get_command("base_velocity")[:, 0]
        act_vx = robot.data.root_lin_vel_b[:, 0]
        vel_errors.append((cmd_vx - act_vx).abs().mean().item())

        done = terminated | truncated
        if done.any():
            obs_reset, _ = env.reset()
            obs_dict["policy"][done] = obs_reset["policy"][done]

        if step % 100 == 0:
            disp = (robot.data.root_pos_w - start_pos).norm(dim=-1).mean().item()
            sr = (~falls).float().mean().item() * 100
            print(f"  [{label}] step {step:4d} | success={sr:.1f}% | "
                  f"vel_err={np.mean(vel_errors):.3f} m/s | disp={disp:.2f} m", flush=True)

    end_pos = robot.data.root_pos_w
    env.close()

    return {
        "success_rate_%":     (~falls).float().mean().item() * 100,
        "mean_vel_error_mps": float(np.mean(vel_errors)),
        "mean_displacement_m": (end_pos - start_pos).norm(dim=-1).mean().item(),
    }


def main():
    device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"

    stock_policy    = torch.jit.load(args_cli.stock_jit).to(device).eval()
    finetuned_policy = torch.jit.load(args_cli.finetuned_jit).to(device).eval()

    print(f"\n[TerrainPilot W3] Evaluating on slope+payload terrain "
          f"({args_cli.num_envs} envs × {args_cli.n_steps} steps)\n", flush=True)

    def make_cfg():
        cfg = UnitreeGo2SlopePayloadEnvCfg_EVAL()
        cfg.scene.num_envs = args_cli.num_envs
        cfg.sim.device = device
        return cfg

    print("[TerrainPilot W3] --- Stock policy (flat terrain, no payload) ---", flush=True)
    stock_m = run_eval("stock",    stock_policy,    make_cfg(), args_cli.n_steps, device)

    print("\n[TerrainPilot W3] --- Fine-tuned policy (slope + payload) ---", flush=True)
    ft_m    = run_eval("finetune", finetuned_policy, make_cfg(), args_cli.n_steps, device)

    print("\n" + "="*62, flush=True)
    print(f"  {'Metric':<28} {'Stock':>12} {'Fine-tuned':>12}", flush=True)
    print("-"*62, flush=True)
    for k in stock_m:
        s, f = stock_m[k], ft_m[k]
        arrow = "↑" if f > s else "↓"
        print(f"  {k:<28} {s:>12.3f} {f:>12.3f}  {arrow}", flush=True)
    print("="*62, flush=True)
    print("\n[TerrainPilot W3] evaluate.py complete ✓", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
