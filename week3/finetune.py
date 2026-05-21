"""
Day 5-6: Fine-tune the stock Go2 policy on sloped terrain with payload.

Loads the Week 2 checkpoint, swaps in the slope+payload env, and trains
for N iterations (default 1000). Saves checkpoints every 200 iterations.

Run:
    cd ~/work/isacc/IsaacLab
    ./isaaclab.sh -p ../TerrainPilot/week3/finetune.py --headless \
        --resume_checkpoint ../logs/rsl_rl/unitree_go2_flat/2026-05-18_08-19-43/model_1499.pt \
        --max_iterations 1000
"""

import argparse, os, sys
from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../IsaacLab/scripts/reinforcement_learning/rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--resume_checkpoint", type=str, required=True,
                    help="Path to Week 2 model_1499.pt checkpoint to fine-tune from.")
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--log_dir", type=str,
                    default="logs/rsl_rl/go2_slope_payload",
                    help="Directory to save fine-tuned checkpoints.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import importlib.metadata as metadata

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

# import our custom env (not a registered gym task — we use it directly)
sys.path.insert(0, os.path.dirname(__file__))
from terrain_payload_env_cfg import UnitreeGo2SlopePayloadEnvCfg  # noqa: E402

installed_rslrl = metadata.version("rsl-rl-lib")
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # swap env config for our slope+payload version
    slope_cfg = UnitreeGo2SlopePayloadEnvCfg()
    slope_cfg.scene.num_envs = args_cli.num_envs
    slope_cfg.sim.device = args_cli.device or "cuda:0"

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rslrl)

    # timestamped output dir
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join(args_cli.log_dir, run_ts))
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n[TerrainPilot W3] Fine-tuning from: {args_cli.resume_checkpoint}", flush=True)
    print(f"[TerrainPilot W3] Output:           {log_dir}", flush=True)
    print(f"[TerrainPilot W3] Iterations:       {args_cli.max_iterations}", flush=True)
    print(f"[TerrainPilot W3] Envs:             {slope_cfg.scene.num_envs}", flush=True)

    env = gym.make(TASK, cfg=slope_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # load Week 2 weights
    runner.load(args_cli.resume_checkpoint)
    print("[TerrainPilot W3] Checkpoint loaded, starting fine-tune...\n", flush=True)

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    print(f"\n[TerrainPilot W3] Fine-tuning complete. Checkpoints in: {log_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
