"""
Day 3-5: Train LeWM on Go2 bamboo-forest trajectories.

Self-contained — bypasses stable_worldmodel.data.load_dataset (not in
installed version) and uses HDF5Dataset directly instead.

Run:
    source ~/work/isacc/env_lewm/bin/activate
    python TerrainPilot/week5/train_go2_lewm.py --epochs 100

Output: TerrainPilot/week5/checkpoints/go2_lewm_best.ckpt
"""

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "le-wm"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf

import stable_worldmodel as swm

from jepa import JEPA
from module import SIGReg, ARPredictor, Embedder, MLP

CACHE_DIR  = os.path.expanduser("~/.stable_worldmodel")
CKPT_DIR   = os.path.join(os.path.dirname(__file__), "checkpoints")

IMG_SIZE    = 224
EMBED_DIM   = 192
HISTORY     = 3      # ctx_len: history frames fed to predictor
N_PREDS     = 1      # predict 1 step ahead
ACTION_DIM  = 3      # [vx, vy, yaw]
SIGREG_W    = 0.09


def build_model(device):
    encoder = instantiate({
        "_target_": "stable_pretraining.backbone.utils.vit_hf",
        "size": "tiny", "patch_size": 14, "image_size": IMG_SIZE,
        "pretrained": False, "use_mask_token": False,
    })
    predictor = ARPredictor(
        num_frames=HISTORY, input_dim=EMBED_DIM, hidden_dim=EMBED_DIM,
        output_dim=EMBED_DIM, depth=6, heads=16, mlp_dim=2048,
        dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=ACTION_DIM, emb_dim=EMBED_DIM)
    projector = MLP(input_dim=EMBED_DIM, output_dim=EMBED_DIM, hidden_dim=2048,
                    norm_fn=nn.BatchNorm1d)
    pred_proj = MLP(input_dim=EMBED_DIM, output_dim=EMBED_DIM, hidden_dim=2048,
                    norm_fn=nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj).to(device)
    return model


def build_dataset():
    img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def transform(batch):
        # pixels: (T, 3, H, W) uint8 → float normalized
        batch["pixels"] = img_transform(batch["pixels"])
        batch["action"] = torch.nan_to_num(batch["action"].float(), 0.0)
        return batch

    dataset = swm.data.HDF5Dataset(
        name="go2_bamboo",
        frameskip=1,
        num_steps=HISTORY + N_PREDS,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        transform=transform,
        cache_dir=CACHE_DIR,
    )
    return dataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[LeWM] Device: {device}")

    # Dataset
    dataset = build_dataset()
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

    print(f"[LeWM] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Model
    model  = build_model(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[LeWM] Parameters: {n_params:.1f}M\n")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(sigreg.parameters()),
        lr=5e-5, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train(); sigreg.train()
        train_pred, train_sig, train_total, n_batches = 0., 0., 0., 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out  = model.encode(batch)
            emb, act_emb = out["emb"], out["act_emb"]   # (B, T, D)

            ctx_emb = emb[:, :HISTORY]
            ctx_act = act_emb[:, :HISTORY]
            tgt_emb = emb[:, N_PREDS:]

            pred_emb  = model.predict(ctx_emb, ctx_act)
            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            sig_loss  = sigreg(emb.transpose(0, 1))
            loss      = pred_loss + SIGREG_W * sig_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_pred  += pred_loss.item()
            train_sig   += sig_loss.item()
            train_total += loss.item()
            n_batches   += 1

        scheduler.step()

        # --- val ---
        model.eval(); sigreg.eval()
        val_pred, val_total, v_batches = 0., 0., 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out  = model.encode(batch)
                emb, act_emb = out["emb"], out["act_emb"]
                ctx_emb = emb[:, :HISTORY]
                ctx_act = act_emb[:, :HISTORY]
                tgt_emb = emb[:, N_PREDS:]
                pred_emb  = model.predict(ctx_emb, ctx_act)
                pred_loss = (pred_emb - tgt_emb).pow(2).mean()
                sig_loss  = sigreg(emb.transpose(0, 1))
                val_pred  += pred_loss.item()
                val_total += (pred_loss + SIGREG_W * sig_loss).item()
                v_batches += 1

        t_pred = train_pred / n_batches
        t_sig  = train_sig  / n_batches
        v_pred = val_pred   / v_batches
        lr     = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:4d}/{args.epochs} | "
              f"train pred={t_pred:.4f} sig={t_sig:.4f} | "
              f"val pred={v_pred:.4f} | lr={lr:.2e}", flush=True)

        if v_pred < best_val:
            best_val = v_pred
            ckpt_path = os.path.join(CKPT_DIR, "go2_lewm_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "sigreg_state": sigreg.state_dict(),
                "val_pred_loss": best_val,
            }, ckpt_path)
            print(f"  ↳ new best val={best_val:.4f}, saved {ckpt_path}", flush=True)

    print(f"\n[LeWM] Training complete. Best val pred_loss: {best_val:.4f}")
    print(f"[LeWM] Checkpoint: {os.path.join(CKPT_DIR, 'go2_lewm_best.pt')}")
    print("[TerrainPilot W5] train_go2_lewm.py complete ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    train(args)
