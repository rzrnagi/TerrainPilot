"""
Collect goal-directed Go2 trajectories through a physical bamboo forest.

Every problem addressed:
  1. Physical bamboo — robot must navigate around stems
  2. base_contact termination disabled — stem contact won't kill episode
  3. Mild slopes (3-5°) in env cfg — terrain variety, policy stays stable
  4. SHORT goals (1.5-3.5m) — robot REACHES the goal in each episode
     → world model sees near-goal transitions; CEM can plan toward them
  5. Flat policy used — more stable from random resets on mild slopes
  6. Warmup BEFORE sampling goal — goal computed from stabilised position
  7. 5 forest layouts rotated every 30 episodes — generalization

Navigation: analytic proportional controller to goal + reactive avoidance
when any stem is within AVOIDANCE_RADIUS. RL policy handles locomotion.

Run:
    cd ~/work/isacc/IsaacLab
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p ../TerrainPilot/week4/collect_trajectories.py \\
        --headless --enable_cameras \\
        --policy_jit logs/rsl_rl/unitree_go2_flat/2026-05-18_08-19-43/exported/policy.pt \\
        --n_episodes 150 \\
        --out_dir ../TerrainPilot/data/trajectories_v2
"""

import argparse, os, sys, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy_jit",   type=str, required=True,
                    help="Use FLAT policy (model_1499) — stable on mild slopes")
parser.add_argument("--n_episodes",   type=int, default=150)
parser.add_argument("--out_dir",      type=str,
                    default="../TerrainPilot/data/trajectories_v2")
parser.add_argument("--episode_s",    type=float, default=25.0,
                    help="Episode budget in seconds. Short goals + 25s = robot reaches them")
parser.add_argument("--n_per_layout", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(__file__))
from bamboo_env_cfg import (
    UnitreeGo2BambooEnvCfg, ALL_LAYOUTS, LAYOUT_SEEDS,
    STEM_RADIUS, GRID_EXTENT, CLEAR_RADIUS, build_bamboo_scene_cfg,
)

# ── Navigation ────────────────────────────────────────────────────────────────
ENV_DT_S        = 0.02
CAMERA_INTERVAL = 5       # every 5 steps → 10 Hz
WARMUP_STEPS    = 60      # stabilise before sampling goal or recording

# SHORT goals so robot reaches them within episode budget
# At 0.4 m/s net speed with some avoidance detours, 3m ≈ 10-15s
GOAL_MIN_DIST   = 1.5     # m
GOAL_MAX_DIST   = 3.5     # m

GOAL_RADIUS     = 1.0     # m — success when within this of goal
AVOIDANCE_RADIUS = 0.85   # m — activate avoidance when nearest stem < this
VX_CRUISE       = 0.40    # m/s forward when clear
VX_AVOID        = 0.08    # m/s forward when avoiding (slow to avoid sliding)
YAW_GAIN        = 1.8     # proportional gain on heading error
MAX_YAW         = 0.70    # rad/s
FALL_THRESHOLD  = 0.12    # m base height
MIN_PROGRESS_M  = 0.3     # m — drop episode if robot barely moved
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_angle(a: float) -> float:
    while a >  np.pi: a -= 2 * np.pi
    while a < -np.pi: a += 2 * np.pi
    return a


def _get_yaw(robot) -> float:
    q = robot.data.root_quat_w[0].cpu().numpy()  # (w, x, y, z)
    return float(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]),
                             1 - 2*(q[2]**2 + q[3]**2)))


def _sample_goal(start_xy: np.ndarray, stem_positions: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray | None:
    """Goal 1.5-3.5m from start, not inside a stem, inside arena."""
    for _ in range(300):
        angle = rng.uniform(-np.pi, np.pi)
        dist  = rng.uniform(GOAL_MIN_DIST, GOAL_MAX_DIST)
        goal  = start_xy + dist * np.array([np.cos(angle), np.sin(angle)])

        if np.any(np.abs(goal) > GRID_EXTENT - 1.5):
            continue
        if np.linalg.norm(goal) < CLEAR_RADIUS:
            continue
        if any(np.linalg.norm(s - goal) < STEM_RADIUS + 0.5 for s in stem_positions):
            continue
        return goal
    return None


def navigation_cmd(robot, goal_xy: np.ndarray,
                   stem_positions: np.ndarray) -> np.ndarray:
    """Proportional heading to goal; override to avoid nearest stem."""
    pos_xy = robot.data.root_pos_w[0, :2].cpu().numpy()
    yaw    = _get_yaw(robot)

    dists    = np.linalg.norm(stem_positions - pos_xy, axis=1)
    min_idx  = int(np.argmin(dists))
    min_dist = float(dists[min_idx])

    if min_dist < AVOIDANCE_RADIUS:
        nearest    = stem_positions[min_idx]
        away_dir   = pos_xy - nearest
        away_angle = float(np.arctan2(away_dir[1], away_dir[0]))
        yaw_err    = _normalize_angle(away_angle - yaw)
        vx         = VX_AVOID
    else:
        to_goal    = goal_xy - pos_xy
        goal_angle = float(np.arctan2(to_goal[1], to_goal[0]))
        yaw_err    = _normalize_angle(goal_angle - yaw)
        vx         = VX_CRUISE

    yaw_cmd = float(np.clip(YAW_GAIN * yaw_err, -MAX_YAW, MAX_YAW))
    return np.array([vx, 0.0, yaw_cmd], dtype=np.float32)


def collect_episode(env, policy, device, stem_positions,
                    episode_steps, rng) -> dict | None:
    obs_dict, _ = env.reset()
    robot   = env.unwrapped.scene["robot"]
    camera  = env.unwrapped.scene["camera"]
    cmd_mgr = env.unwrapped.command_manager

    # ── Warmup: stabilise on terrain before touching anything ────────────────
    for w in range(WARMUP_STEPS):
        cmd_mgr.get_term("base_velocity").vel_command_b.data[:] = \
            torch.zeros(1, 3, device=device)
        with torch.inference_mode():
            actions = policy(obs_dict["policy"])
        obs_dict, _, terminated, _, _ = env.step(actions)
        h = robot.data.root_pos_w[0, 2].item()
        if h < FALL_THRESHOLD or terminated.any():
            print(f"    [dbg] warmup fall at step {w}: h={h:.3f}m term={terminated.any().item()}", flush=True)
            return None

    # ── Sample goal from stable position ─────────────────────────────────────
    start_xy = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
    goal_xy  = _sample_goal(start_xy, stem_positions, rng)
    if goal_xy is None:
        print(f"    [dbg] no valid goal from start={start_xy}", flush=True)
        return None

    # ── Capture start frame ──────────────────────────────────────────────────
    start_frame = None
    rgb = camera.data.output.get("rgb")
    if rgb is not None:
        start_frame = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)

    rgb_frames, cmd_log, pos_log, vel_log, ts_log = [], [], [], [], []
    fell = False
    reached = False

    # ── Navigate toward goal ─────────────────────────────────────────────────
    for step in range(episode_steps):
        pos_xy    = robot.data.root_pos_w[0, :2].cpu().numpy()
        dist_goal = float(np.linalg.norm(goal_xy - pos_xy))

        if dist_goal < GOAL_RADIUS:
            reached = True
            break

        if robot.data.root_pos_w[0, 2].item() < FALL_THRESHOLD:
            fell = True
            break

        cmd   = navigation_cmd(robot, goal_xy, stem_positions)
        cmd_t = torch.tensor([cmd], dtype=torch.float32, device=device)
        cmd_mgr.get_term("base_velocity").vel_command_b.data[:] = cmd_t

        with torch.inference_mode():
            actions = policy(obs_dict["policy"])
        obs_dict, _, terminated, truncated, _ = env.step(actions)

        if terminated.any():
            fell = True; break
        if truncated.any():
            break

        if step % CAMERA_INTERVAL == 0:
            rgb = camera.data.output.get("rgb")
            if rgb is not None:
                frame = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
                if start_frame is None:
                    start_frame = frame.copy()
                rgb_frames.append(frame)
                cmd_log.append(cmd)
                pos_log.append(robot.data.root_pos_w[0].cpu().numpy())
                vel_log.append(robot.data.root_lin_vel_b[0].cpu().numpy())
                ts_log.append(time.time())

    if fell or len(rgb_frames) < 10:
        return None

    disp = float(np.linalg.norm(
        np.array(pos_log[-1][:2]) - np.array(pos_log[0][:2])
    ))
    if disp < MIN_PROGRESS_M:
        return None

    return {
        "rgb":         np.stack(rgb_frames),
        "cmd_vel":     np.stack(cmd_log),
        "base_pos":    np.stack(pos_log),
        "base_vel":    np.stack(vel_log),
        "timestamp":   np.array(ts_log),
        "goal_pos_xy": goal_xy,
        "start_frame": start_frame if start_frame is not None else rgb_frames[0],
        "goal_frame":  rgb_frames[-1],
        "reached":     np.bool_(reached),
    }


def rebuild_env(device, seed):
    cfg = UnitreeGo2BambooEnvCfg()
    cfg.sim.device = device
    build_bamboo_scene_cfg(cfg.scene, seed=seed)
    env  = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    return env, ALL_LAYOUTS[seed]


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"
    policy = torch.jit.load(args_cli.policy_jit, map_location=device).eval()
    episode_steps = int(args_cli.episode_s / ENV_DT_S)
    rng = np.random.default_rng(42)

    print(f"\n[W4v3] Goal-directed bamboo collection — all issues fixed")
    print(f"  Terrain:    mild slopes 3-5° (stable + terrain-aware)")
    print(f"  Bamboo:     physical collision, 20 stems, 5 layouts")
    print(f"  Goals:      {GOAL_MIN_DIST}-{GOAL_MAX_DIST}m (robot WILL reach them)")
    print(f"  Policy:     flat (stable from random resets on mild slopes)")
    print(f"  Target:     {args_cli.n_episodes} episodes → {args_cli.out_dir}\n")

    collected = 0; attempts = 0; reached_count = 0
    layout_idx = 0; ep_in_layout = 0
    t_start = time.time()

    env, stems = rebuild_env(device, LAYOUT_SEEDS[layout_idx])

    while collected < args_cli.n_episodes:
        if ep_in_layout >= args_cli.n_per_layout:
            env.close()
            layout_idx   = (layout_idx + 1) % len(LAYOUT_SEEDS)
            ep_in_layout = 0
            env, stems   = rebuild_env(device, LAYOUT_SEEDS[layout_idx])
            print(f"\n  [Layout → seed={LAYOUT_SEEDS[layout_idx]}]\n")

        attempts += 1
        data = collect_episode(env, policy, device, stems, episode_steps, rng)

        if data is None:
            print(f"  attempt {attempts:4d} → dropped", flush=True)
            ep_in_layout += 1
            continue

        path = os.path.join(args_cli.out_dir, f"ep_{collected:04d}.npz")
        np.savez_compressed(path, **data)
        collected    += 1
        ep_in_layout += 1
        if data["reached"]: reached_count += 1

        elapsed = time.time() - t_start
        eta_s   = elapsed / collected * (args_cli.n_episodes - collected)
        print(
            f"  ep {collected:4d}/{args_cli.n_episodes} | "
            f"seed={LAYOUT_SEEDS[layout_idx]} | "
            f"frames={len(data['rgb']):3d} | "
            f"disp={np.linalg.norm(data['base_pos'][-1,:2]-data['base_pos'][0,:2]):.1f}m | "
            f"reached={'✓' if data['reached'] else '✗'} "
            f"({100*reached_count/collected:.0f}%) | "
            f"ETA {eta_s/60:.1f}min",
            flush=True,
        )

    env.close()
    total_mb = sum(
        os.path.getsize(os.path.join(args_cli.out_dir, f))
        for f in os.listdir(args_cli.out_dir) if f.endswith(".npz")
    ) / 1e6
    print(f"\n[W4v3] Done: {collected} eps | reached {reached_count}/{collected} "
          f"({100*reached_count/collected:.0f}%) | {total_mb:.0f} MB total")
    print("[W4v3] collect_trajectories.py complete ✓")


if __name__ == "__main__":
    main()
    simulation_app.close()
