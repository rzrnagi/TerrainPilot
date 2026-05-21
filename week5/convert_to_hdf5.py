"""
Day 1: Convert Week 4 .npz trajectories to LeWM HDF5 format.

HDF5 schema expected by stable_worldmodel.data.HDF5Dataset:
  ep_len    (N_ep,)              — number of steps in each episode
  ep_offset (N_ep,)              — cumulative start index of each episode
  pixels    (N_total, H, W, 3)  — uint8 RGB frames
  action    (N_total, 3)         — float32 cmd_vel [vx, vy, yaw]

Run:
    python week5/convert_to_hdf5.py \
        --data_dir TerrainPilot/data/trajectories \
        --out ~/.stable_worldmodel/go2_bamboo.h5 \
        --min_frames 50
"""

import argparse, glob, os
import numpy as np
import h5py
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default="data/trajectories")
    p.add_argument("--out",        type=str, default=os.path.expanduser("~/.stable_worldmodel/go2_bamboo.h5"))
    p.add_argument("--min_frames", type=int, default=50,  help="Drop episodes shorter than this")
    p.add_argument("--val_frac",   type=float, default=0.1, help="Fraction saved as val split (separate file)")
    return p.parse_args()


def _scan_episodes(files: list, min_frames: int):
    """First pass: collect lengths without loading pixels."""
    ep_lens, valid_files = [], []
    for path in files:
        with np.load(path) as d:
            T = len(d["rgb"])
        if T >= min_frames:
            ep_lens.append(T)
            valid_files.append(path)
    return valid_files, np.array(ep_lens, dtype=np.int64)


def _write_hdf5_chunked(path: str, files: list, ep_lens: np.ndarray):
    """Write HDF5 episode-by-episode to avoid loading everything into RAM."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_total = int(ep_lens.sum())
    ep_offset = np.concatenate([[0], np.cumsum(ep_lens)[:-1]]).astype(np.int64)

    # peek at first episode for shape
    with np.load(files[0]) as d:
        H, W, C = d["rgb"].shape[1:]

    with h5py.File(path, "w") as f:
        f.create_dataset("ep_len",    data=ep_lens,   compression="lzf")
        f.create_dataset("ep_offset", data=ep_offset, compression="lzf")
        pix_ds = f.create_dataset("pixels", shape=(n_total, H, W, C),
                                  dtype=np.uint8, compression="lzf",
                                  chunks=(1, H, W, C))
        act_ds = f.create_dataset("action", shape=(n_total, 3),
                                  dtype=np.float32, compression="lzf")

        cursor = 0
        for i, (fpath, T) in enumerate(zip(files, ep_lens)):
            with np.load(fpath) as d:
                pix_ds[cursor:cursor+T] = d["rgb"]
                act_ds[cursor:cursor+T] = d["cmd_vel"].astype(np.float32)
            cursor += T
            if (i + 1) % 20 == 0:
                print(f"  written {i+1}/{len(files)} episodes...", flush=True)

    size_mb = os.path.getsize(path) / 1e6
    print(f"  → {path}  ({len(files)} eps, {n_total} steps, {size_mb:.0f} MB)")
    return path


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "ep_*.npz")))
    if not files:
        print(f"No .npz files found in {args.data_dir}"); return

    print(f"Found {len(files)} episodes — scanning (no pixel load)...")
    valid_files, ep_lens = _scan_episodes(files, args.min_frames)
    dropped = len(files) - len(valid_files)
    print(f"Kept {len(valid_files)} episodes, dropped {dropped} (< {args.min_frames} frames)")

    # train/val split by episode index
    n_val   = max(1, int(len(ep_lens) * args.val_frac))
    n_train = len(ep_lens) - n_val

    print(f"\nWriting train HDF5 ({n_train} episodes)...")
    _write_hdf5_chunked(args.out, valid_files[:n_train], ep_lens[:n_train])

    val_out = args.out.replace(".h5", "_val.h5")
    print(f"Writing val HDF5 ({n_val} episodes)...")
    _write_hdf5_chunked(val_out, valid_files[n_train:], ep_lens[n_train:])

    print("\n[TerrainPilot W5] convert_to_hdf5.py complete ✓")
    print(f"  Train: {args.out}")
    print(f"  Val:   {val_out}")
    print(f"  Action dim: 3  [vx_mps, vy_mps, yaw_rps]")


if __name__ == "__main__":
    main()
