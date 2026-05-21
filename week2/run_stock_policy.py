"""
Day 1-2: Load stock Go2 checkpoint, run inference, verify robot walks.

Loads the latest checkpoint from Isaac Lab's training logs, wraps the env
with rsl_rl's VecEnvWrapper, and runs the policy for N_STEPS steps.
Prints displacement at the end to confirm locomotion.

Run:
    cd ~/work/isacc/IsaacLab
    ./isaaclab.sh -p ../TerrainPilot/week2/run_stock_policy.py --headless

    # Specify a checkpoint explicitly:
    ./isaaclab.sh -p ../TerrainPilot/week2/run_stock_policy.py --headless \
        --checkpoint logs/rsl_rl/unitree_go2_flat/<run>/model_1499.pt
"""

import argparse, os, sys
from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../IsaacLab/scripts/reinforcement_learning/rsl_rl"))
import cli_args  # noqa: E402 — from Isaac Lab scripts dir

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--n_steps",  type=int, default=500)
cli_args.add_rsl_rl_args(parser)  # adds --checkpoint among others
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import importlib.metadata as metadata
from packaging import version

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
installed_rslrl = metadata.version("rsl-rl-lib")


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rslrl)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or "cuda:0"

    # resolve checkpoint path
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"\n[TerrainPilot W2] Loading checkpoint: {resume_path}", flush=True)

    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export JIT + ONNX alongside the checkpoint
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_rslrl) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
    print(f"[TerrainPilot W2] Exported JIT + ONNX → {export_dir}", flush=True)

    obs = env.get_observations()
    start_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].clone()

    print(f"[TerrainPilot W2] Running policy for {args_cli.n_steps} steps...", flush=True)
    for step in range(args_cli.n_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_rslrl) >= version.parse("4.0.0"):
                policy.reset(dones)

        if step % 100 == 0:
            pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
            vel = env.unwrapped.scene["robot"].data.root_lin_vel_b[0]
            disp = (pos - start_pos).norm().item()
            print(f"  step {step:4d} | pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] | "
                  f"vel_x={vel[0]:.3f} m/s | displacement={disp:.3f} m", flush=True)

    end_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    total_disp = (end_pos - start_pos).norm().item()
    print(f"\n[TerrainPilot W2] Total displacement: {total_disp:.3f} m", flush=True)
    print(f"[TerrainPilot W2] run_stock_policy.py complete ✓", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
