"""
Day 6-7: Inspect collected trajectory dataset.

Prints per-episode stats, flags corrupted/short episodes, checks
action-observation alignment, and saves thumbnail strips.

Run (no sim needed):
    python ../TerrainPilot/week4/inspect_data.py \
        --data_dir ../TerrainPilot/data/trajectories \
        --save_thumbs
"""

import argparse, os, glob
import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="data/trajectories")
    p.add_argument("--save_thumbs", action="store_true",
                   help="Save a thumbnail strip for each episode")
    p.add_argument("--thumb_dir",   type=str, default=None,
                   help="Where to save thumbs (default: data_dir/thumbs)")
    p.add_argument("--min_frames",  type=int, default=50,
                   help="Minimum frames for a valid episode")
    p.add_argument("--max_vel_err", type=float, default=5.0,
                   help="Flag episodes where |cmd_vx - act_vx| > this")
    return p.parse_args()


def load_episode(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def episode_stats(data: dict, path: str, min_frames: int, max_vel_err: float) -> dict:
    rgb  = data["rgb"]         # (T, H, W, 3)
    cmd  = data["cmd_vel"]     # (T, 3)
    pos  = data["base_pos"]    # (T, 3)
    vel  = data["base_vel"]    # (T, 3)
    ts   = data["timestamp"]   # (T,)
    T    = len(rgb)

    issues = []
    if T < min_frames:
        issues.append(f"short ({T} frames)")
    if np.isnan(rgb).any() or np.isnan(cmd).any():
        issues.append("NaN values")
    if rgb.max() == 0:
        issues.append("black frames")

    # alignment check: timestamp gaps
    dt = np.diff(ts)
    if len(dt) > 0 and (dt.max() > 0.5 or dt.min() < 0):
        issues.append(f"bad timestamps (dt_max={dt.max():.3f}s)")

    disp = float(np.linalg.norm(pos[-1] - pos[0])) if T > 1 else 0.0
    vx_err = float(np.abs(cmd[:, 0] - vel[:, 0]).mean())

    if vx_err > max_vel_err:
        issues.append(f"high vel_err ({vx_err:.2f} m/s)")

    return {
        "file":      os.path.basename(path),
        "frames":    T,
        "duration_s": float(ts[-1] - ts[0]) if T > 1 else 0.0,
        "disp_m":    disp,
        "cmd_vx_mean": float(cmd[:, 0].mean()),
        "vx_err_mps": vx_err,
        "issues":    issues,
        "ok":        len(issues) == 0,
    }


def save_thumbnail_strip(rgb: np.ndarray, path: str, n_thumbs: int = 8):
    """Save evenly-spaced frames as a horizontal strip."""
    T = len(rgb)
    idxs = np.linspace(0, T - 1, n_thumbs, dtype=int)
    frames = [cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR) for i in idxs]
    strip = np.concatenate(frames, axis=1)
    cv2.imwrite(path, strip)


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "ep_*.npz")))

    if not files:
        print(f"[Inspect] No .npz files found in {args.data_dir}")
        return

    thumb_dir = args.thumb_dir or os.path.join(args.data_dir, "thumbs")
    if args.save_thumbs:
        os.makedirs(thumb_dir, exist_ok=True)

    all_stats = []
    bad_files = []

    print(f"\n[TerrainPilot W4] Inspecting {len(files)} episodes in {args.data_dir}\n")

    for path in files:
        try:
            data = load_episode(path)
            stats = episode_stats(data, path, args.min_frames, args.max_vel_err)
            all_stats.append(stats)

            if args.save_thumbs and stats["ok"]:
                save_thumbnail_strip(
                    data["rgb"],
                    os.path.join(thumb_dir, stats["file"].replace(".npz", ".jpg")),
                )

            status = "✓" if stats["ok"] else f"✗ {', '.join(stats['issues'])}"
            print(f"  {stats['file']}  frames={stats['frames']:4d}  "
                  f"disp={stats['disp_m']:.1f}m  "
                  f"vel_err={stats['vx_err_mps']:.3f}  {status}")

            if not stats["ok"]:
                bad_files.append(path)

        except Exception as e:
            print(f"  {os.path.basename(path)}  ERROR: {e}")
            bad_files.append(path)

    # Summary
    ok  = [s for s in all_stats if s["ok"]]
    print(f"\n{'='*60}")
    print(f"  Total episodes:   {len(all_stats)}")
    print(f"  Valid episodes:   {len(ok)}")
    print(f"  Bad / flagged:    {len(bad_files)}")
    if ok:
        frames    = [s["frames"]    for s in ok]
        disps     = [s["disp_m"]    for s in ok]
        vel_errs  = [s["vx_err_mps"]for s in ok]
        durs      = [s["duration_s"]for s in ok]
        print(f"  Mean duration:    {np.mean(durs):.1f} s  (range {min(durs):.1f}–{max(durs):.1f})")
        print(f"  Mean frames:      {np.mean(frames):.0f}  (range {min(frames)}–{max(frames)})")
        print(f"  Mean disp:        {np.mean(disps):.1f} m  (range {min(disps):.1f}–{max(disps):.1f})")
        print(f"  Mean vel_err:     {np.mean(vel_errs):.3f} m/s")
    print(f"{'='*60}")

    if bad_files:
        print(f"\n  Flagged for review / removal:")
        for f in bad_files:
            print(f"    {f}")

    print("\n[TerrainPilot W4] inspect_data.py complete ✓")


if __name__ == "__main__":
    main()
