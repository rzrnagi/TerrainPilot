"""
Train a high-success-rate locomotion policy for bamboo data collection.

Uses 4096 parallel envs with push randomization and mild slopes.
Loads from the Week 2 flat checkpoint for fast convergence.
Target: >90% episode completion rate on the bamboo collection env.

Run:
    cd ~/work/isacc/IsaacLab
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p ../TerrainPilot/week3/train_bamboo_policy.py \\
        --headless \\
        --resume_checkpoint logs/rsl_rl/unitree_go2_flat/2026-05-18_08-19-43/model_1499.pt \\
        --max_iterations 2000
"""

import argparse, os, sys
from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                "../../IsaacLab/scripts/reinforcement_learning/rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--resume_checkpoint", type=str, required=True)
parser.add_argument("--max_iterations",    type=int, default=2000)
parser.add_argument("--log_dir", type=str,
                    default="logs/rsl_rl/go2_bamboo_locomotion")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

sys.path.insert(0, os.path.dirname(__file__))
from bamboo_locomotion_env_cfg import BambooLocomotionEnvCfg  # noqa: E402

installed_rslrl = metadata.version("rsl-rl-lib")
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    cfg = BambooLocomotionEnvCfg()
    cfg.sim.device = args_cli.device or "cuda:0"

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rslrl)

    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join(args_cli.log_dir, run_ts))
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n[TerrainPilot] Bamboo locomotion policy training")
    print(f"  Resume:     {args_cli.resume_checkpoint}")
    print(f"  Envs:       {cfg.scene.num_envs} (4096 parallel)")
    print(f"  Iterations: {args_cli.max_iterations}")
    print(f"  Output:     {log_dir}")
    print(f"  Push robot: every 3-8s (bamboo collision simulation)")
    print(f"  Terrain:    mild slopes 1-5° with curriculum\n")

    env = gym.make(TASK, cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(),
                            log_dir=log_dir, device=agent_cfg.device)
    runner.load(args_cli.resume_checkpoint)

    print("[TerrainPilot] Checkpoint loaded — starting training...\n", flush=True)
    runner.learn(num_learning_iterations=args_cli.max_iterations,
                 init_at_random_ep_len=True)

    # Export for collection
    from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
    from packaging import version
    export_dir = os.path.join(log_dir, "exported")
    if version.parse(installed_rslrl) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_dir,  filename="policy.pt")
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")

    print(f"\n[TerrainPilot] Training complete. Policy exported → {export_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
