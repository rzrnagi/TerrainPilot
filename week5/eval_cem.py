"""
Day 6-7: Evaluate LeWM CEM planner on held-out Go2 trajectories.

Loads a trained LeWM checkpoint, picks random (start, goal) image pairs
from the val set, runs CEM planning to find action sequences, and reports
whether the predicted trajectory reaches the goal embedding.

Qualitative check: the plan says "CEM planning produces action sequences
that are qualitatively sensible — not necessarily perfect, not random noise."

Run:
    source env_lewm/bin/activate
    python TerrainPilot/week5/eval_cem.py \
        --checkpoint ~/.stable_worldmodel/go2_lewm/<id>_object.ckpt \
        --val_h5 ~/.stable_worldmodel/go2_bamboo_val.h5 \
        --n_pairs 20
"""

import argparse, os
import numpy as np
import torch
import h5py
from torchvision import transforms

# LeWM imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "le-wm"))
from jepa import JEPA


# CEM hyperparameters
CEM_ITERS     = 5
CEM_SAMPLES   = 512
CEM_ELITE     = 64
HORIZON       = 7       # plan 7 steps ahead (~0.7s at 10Hz)
ACTION_DIM    = 3       # [vx, vy, yaw]
ACTION_LO     = torch.tensor([0.0, -0.2, -0.6])
ACTION_HI     = torch.tensor([0.8,  0.2,  0.6])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--val_h5",     type=str,
                   default=os.path.expanduser("~/.stable_worldmodel/go2_bamboo_val.h5"))
    p.add_argument("--n_pairs",    type=int, default=20)
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


def load_model(ckpt_path: str, device: str) -> JEPA:
    """Load JEPA from a stable_worldmodel Lightning checkpoint."""
    import stable_worldmodel as swm
    # Lightning saves the module as state_dict inside the checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    # The Lightning module wraps JEPA — extract via state_dict key prefix
    state = {k.replace("model.", "", 1): v
             for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    # Rebuild model from config stored in checkpoint
    model_cfg = ckpt.get("hyper_parameters", {}).get("cfg", {}).get("model", {})
    # Fallback: build a default ViT-tiny LeWM
    from hydra import compose, initialize_config_dir
    import os
    cfg_dir = os.path.join(os.path.dirname(__file__), "le-wm/config/train")
    with initialize_config_dir(config_dir=os.path.abspath(cfg_dir), version_base=None):
        cfg = compose("go2_lewm")
    from hydra.utils import instantiate
    model = instantiate(cfg.model, action_encoder={"input_dim": ACTION_DIM}).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def preprocess_frame(frame_np: np.ndarray, img_size: int) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, img_size, img_size) float tensor."""
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return t(frame_np).unsqueeze(0)


def cem_plan(model, ctx_frames, goal_frame, device, img_size):
    """
    CEM planning: find action sequence that minimises dist(pred_emb, goal_emb).

    ctx_frames: list of (H,W,3) uint8 — last `history_size` frames
    goal_frame: (H,W,3) uint8 — target image
    Returns: (HORIZON, ACTION_DIM) best action sequence, final cost
    """
    # Encode context
    ctx_tensors = torch.cat([preprocess_frame(f, img_size) for f in ctx_frames], dim=0)
    ctx_tensors = ctx_tensors.unsqueeze(0).to(device)          # (1, T_ctx, 3, H, W)

    # Encode goal
    goal_tensor = preprocess_frame(goal_frame, img_size).unsqueeze(0).to(device)  # (1,1,3,H,W)

    with torch.no_grad():
        # Encode context frames
        ctx_out = model.encode({"pixels": ctx_tensors.view(-1, 3, img_size, img_size)})
        # ctx_emb: (1, T_ctx, D)
        ctx_emb = ctx_out["emb"].view(1, len(ctx_frames), -1)

        # Encode goal
        goal_out = model.encode({"pixels": goal_tensor.view(-1, 3, img_size, img_size)})
        goal_emb = goal_out["emb"].view(1, 1, -1)              # (1, 1, D)

    # CEM over action sequences (HORIZON, ACTION_DIM)
    lo = ACTION_LO.to(device)
    hi = ACTION_HI.to(device)
    mu  = (lo + hi) / 2
    std = (hi - lo) / 4

    best_cost = float("inf")
    best_seq  = mu.unsqueeze(0).expand(HORIZON, -1)

    for it in range(CEM_ITERS):
        # Sample (S, HORIZON, ACTION_DIM)
        noise = torch.randn(CEM_SAMPLES, HORIZON, ACTION_DIM, device=device)
        seqs  = (mu + std * noise).clamp(lo, hi)              # (S, H, A)

        # Encode actions: (S, H, emb_dim)
        act_flat = seqs.view(-1, ACTION_DIM)
        act_emb  = model.action_encoder(act_flat).view(CEM_SAMPLES, HORIZON, -1)

        # Autoregressive rollout from context for each candidate
        costs = torch.zeros(CEM_SAMPLES, device=device)
        emb_batch = ctx_emb.expand(CEM_SAMPLES, -1, -1).clone()  # (S, T_ctx, D)

        for h in range(HORIZON):
            with torch.no_grad():
                # predict next embedding from last history_size frames
                pred = model.predict(
                    emb_batch[:, -3:],       # (S, 3, D)
                    act_emb[:, h:h+1],       # (S, 1, A_emb)
                )                            # (S, 1, D)
            emb_batch = torch.cat([emb_batch, pred], dim=1)

        # Cost: MSE to goal embedding
        pred_last = emb_batch[:, -1:, :]         # (S, 1, D)
        costs = (pred_last - goal_emb).pow(2).mean(dim=-1).squeeze()  # (S,)

        # Elite update
        elite_idx = costs.topk(CEM_ELITE, largest=False).indices
        elite_seqs = seqs[elite_idx]             # (elite, H, A)
        mu  = elite_seqs.mean(dim=0)
        std = elite_seqs.std(dim=0) + 1e-6

        min_cost = costs[elite_idx].mean().item()
        if min_cost < best_cost:
            best_cost = min_cost
            best_seq  = mu

    return best_seq.cpu().numpy(), best_cost


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n[TerrainPilot W5] Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    print(f"[TerrainPilot W5] Loading val data from {args.val_h5}")
    with h5py.File(args.val_h5, "r") as f:
        pixels    = f["pixels"][:]   # (N, H, W, 3)
        ep_lens   = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

    rng = np.random.default_rng(42)
    history_size = 3

    print(f"\n[TerrainPilot W5] Running CEM planning on {args.n_pairs} (start, goal) pairs\n")
    costs = []

    for i in range(args.n_pairs):
        ep_idx   = rng.integers(len(ep_lens))
        ep_start = int(ep_offset[ep_idx])
        ep_len   = int(ep_lens[ep_idx])
        if ep_len < history_size + 10:
            continue

        # random start (must have history_size preceding frames)
        t_start = rng.integers(history_size, ep_len - 5)
        t_goal  = rng.integers(t_start + 5, ep_len)

        ctx_frames = [pixels[ep_start + t_start - history_size + j] for j in range(history_size)]
        goal_frame = pixels[ep_start + t_goal]

        seq, cost = cem_plan(model, ctx_frames, goal_frame, device, args.img_size)
        costs.append(cost)

        vx_range  = f"{seq[:, 0].min():.2f}–{seq[:, 0].max():.2f}"
        yaw_range = f"{seq[:, 2].min():.2f}–{seq[:, 2].max():.2f}"
        print(f"  pair {i+1:3d} | ep={ep_idx:4d} t={t_start}→{t_goal:4d} | "
              f"cost={cost:.4f} | vx∈[{vx_range}] yaw∈[{yaw_range}]")

    print(f"\n[TerrainPilot W5] Mean planning cost: {np.mean(costs):.4f}")
    print(f"[TerrainPilot W5] Qualitative check: vx and yaw should vary (not all zeros or all max)")
    print("[TerrainPilot W5] eval_cem.py complete ✓")


if __name__ == "__main__":
    main()
