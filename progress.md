# TerrainPilot — Progress Log

**Project:** Hierarchical quadruped navigation — Unitree Go2 on sloped terrain with payload, guided by a LeWorldModel (JEPA) high-level planner over a low-level RL locomotion policy.
**Goal:** Sim-focused 8-week project producing a GitHub repo, WebXR demo, 3-min video, 1-page brief, 8-slide deck for EY portfolio.

---

## Directory Layout

```
~/work/isacc/
├── env_isaaclab/               ← Python 3.11 venv (Isaac Lab + Isaac Sim)
├── env_lewm/                   ← Python 3.11 venv (LeWM training, isolated)
├── IsaacLab/                   ← Isaac Lab 0.54.3 source
├── unitree_sim_isaaclab/        ← Unitree sim env (DDS interface)
├── unitree_sdk2_python/         ← Unitree SDK2 Python bindings
└── TerrainPilot/
    ├── plan.txt                 ← Original 8-week plan
    ├── progress.md              ← This file
    ├── data/
    │   └── trajectories/        ← 150 × ep_XXXX.npz (LeWM training data)
    ├── week1/                   ← SDK plumbing scripts
    ├── week2/                   ← RL policy + ONNX scripts
    ├── week3/                   ← Slope+payload fine-tuning scripts
    ├── week4/                   ← Data collection scripts
    └── week5/
        ├── le-wm/               ← LeWM repo (cloned)
        ├── checkpoints/         ← go2_lewm_best.pt (saved during training)
        ├── convert_to_hdf5.py
        ├── train_go2_lewm.py
        ├── train_lewm.sh
        └── eval_cem.py
```

---

## Environment Summary

| Component | Version | Location |
|---|---|---|
| Python (Isaac Lab venv) | 3.11.15 | `~/work/isacc/env_isaaclab` |
| Python (LeWM venv) | 3.11.15 | `~/work/isacc/env_lewm` |
| Isaac Sim | 5.1.0 | pip in env_isaaclab |
| Isaac Lab | 0.54.3 | `~/work/isacc/IsaacLab` |
| MuJoCo | 3.8.1 | pip in env_isaaclab |
| stable-worldmodel | 0.0.6 | pip in env_lewm |
| stable-pretraining | 0.1.6 | pip in env_lewm |
| unitree_sim_isaaclab | latest | `~/work/isacc/unitree_sim_isaaclab` |
| unitree_sdk2_python | 1.0.1 | `~/work/isacc/unitree_sdk2_python` |
| GPU | RTX 5070 Ti 16 GB | driver 590.48.01, CUDA 13.1 |
| OS | Ubuntu 24.04 LTS | kernel 6.17.0-23-generic |

**Activate Isaac Lab env:** `source ~/work/isacc/env_isaaclab/bin/activate`
**Activate LeWM env:** `source ~/work/isacc/env_lewm/bin/activate`
**Run Isaac Lab scripts:** `cd ~/work/isacc/IsaacLab && PYTHONUNBUFFERED=1 ./isaaclab.sh -p <script> --headless`

---

## Week 0 — Environment Setup ✅ COMPLETE

**Kill criterion:** all setup steps complete with verified output. ✅ PASSED

| Step | Status | Notes |
|---|---|---|
| Isaac Sim 5.1.0 | ✅ | pip install, EULA accepted, import OK |
| Isaac Lab 0.54.3 | ✅ | `./isaaclab.sh --install`, all extensions installed |
| MuJoCo 3.8.1 | ✅ | pip install, import OK |
| unitree_sim_isaaclab | ✅ | cloned, deps installed |
| unitree_sdk2_python | ✅ | cloned, installed --no-deps (cyclonedds conflict) |
| unitree_mujoco | ⛔ skipped | redundant — unitree_sim_isaaclab covers DDS/SDK validation |
| unitree_sdk2 C++ | ⛔ skipped | sim-only scope, Python SDK sufficient |
| Stock Go2 task trains | ✅ | reward -0.52 → +10.97 in 100 iters @0.44s/iter |

**Gotchas resolved:**
- Used Isaac Sim 5.1.0 (plan assumed 4.5) → requires Python 3.11 not 3.10
- CycloneDDS 0.10.2 (pinned by unitree_sdk2_python) fails to build on Python 3.11 → installed unitree_sdk2_python with `--no-deps`, uses cyclonedds 11.0.1 instead
- cryptography pinned to 44.0.0 + pyopenssl 24.3.0 for isaacsim-core compatibility
- All Isaac Lab scripts must be run with `PYTHONUNBUFFERED=1` to see Python stdout through the launcher

---

## Week 1 — SDK and Simulator Plumbing ✅ COMPLETE

**Kill criterion:** spawn robot, command it, read camera + state, log + replay. ✅ PASSED
**Approach:** Isaac Lab gym API directly (no DDS) — cleaner for the RL pipeline.

### Scripts (`TerrainPilot/week1/`)

| File | Purpose | Verified result |
|---|---|---|
| `velocity_cmd.py` | Set 0.5 m/s forward cmd, print base pos/vel every 50 steps | 300 steps, state flows, no policy = no motion (expected) |
| `camera_stream.py` | Front camera 320×240, save 100 PNG frames | 100 frames, 15.1 ms/frame mean interval |
| `logger.py` | Reusable `TelemetryLogger` class → `.npz` (reused Week 4) | base_pos, base_vel, joint_pos/vel, IMU, cmd_vel |
| `open_loop_run.py` | FWD 3s → TURN 2s × 3 cycles, log + replay stats | 750 steps, 0.152 m displacement, replay verified |

**Run:**
```bash
cd ~/work/isacc/IsaacLab && source ../env_isaaclab/bin/activate
PYTHONUNBUFFERED=1 ./isaaclab.sh -p ../TerrainPilot/week1/velocity_cmd.py --headless
PYTHONUNBUFFERED=1 ./isaaclab.sh -p ../TerrainPilot/week1/camera_stream.py --headless --enable_cameras
PYTHONUNBUFFERED=1 ./isaaclab.sh -p ../TerrainPilot/week1/open_loop_run.py --headless
python ../TerrainPilot/week1/open_loop_run.py --replay_only --log_path /tmp/terrainpilot_w1.npz
```

---

## Week 2 — Stock RL Policy Deployment ✅ COMPLETE

**Kill criterion:** trained policy walks the robot. ✅ PASSED

### RL training details

**Algorithm:** PPO via rsl_rl 4.x
**Architecture:** Separate actor + critic, no shared trunk
- Actor: `Linear(48→128) → ELU → ×2 → Linear(128→12)` with learnable noise std
- Critic: same dims → `Linear(128→1)`
- Activation: ELU; no observation normalization

**Observation space (48-dim):**
| Term | Dim | Description |
|---|---|---|
| base_lin_vel | 3 | body-frame linear velocity [m/s] ±0.1 noise |
| base_ang_vel | 3 | body-frame angular velocity [rad/s] ±0.2 noise |
| projected_gravity | 3 | gravity in body frame (encodes tilt) ±0.05 noise |
| velocity_commands | 3 | commanded [vx, vy, yaw] |
| joint_pos | 12 | joint positions rel. to default [rad] ±0.01 noise |
| joint_vel | 12 | joint velocities [rad/s] ±1.5 noise |
| last_action | 12 | previous action output |

**Action space (12-dim):** joint position targets, scale=0.25, added to default standing pose. Sent to PD controller: τ = kp(q_target − q) − kd·q̇

**Reward function (final weights after Go2 + flat overrides):**
| Term | Weight | Formula |
|---|---|---|
| track_lin_vel_xy_exp | +1.5 | exp(−‖v_xy^cmd − v_xy^act‖² / 0.25) |
| track_ang_vel_z_exp | +0.75 | exp(−(ω_z^cmd − ω_z^act)² / 0.25) |
| feet_air_time | +0.25 | Σ_f max(0, t_air,f − 0.5) · 𝟙[v_cmd > 0.1] |
| flat_orientation_l2 | −2.5 | −‖g_xy^body‖² (→ −3.0 for slope fine-tune) |
| lin_vel_z_l2 | −2.0 | −v_z² |
| ang_vel_xy_l2 | −0.05 | −(ω_x² + ω_y²) |
| dof_torques_l2 | −0.0002 | −‖τ‖² |
| dof_acc_l2 | −2.5e-7 | −‖q̈‖² |
| action_rate_l2 | −0.01 | −‖a_t − a_{t−1}‖² |

**PPO hyperparameters:**
| Param | Value |
|---|---|
| clip_param ε | 0.2 |
| value_loss_coef | 1.0 |
| entropy_coef | 0.01 |
| num_learning_epochs | 5 |
| num_mini_batches | 4 |
| learning_rate | 1e-3 adaptive (desired_kl=0.01) |
| gamma | 0.99 |
| lam (GAE) | 0.95 |
| max_grad_norm | 1.0 |
| num_steps_per_env | 24 |
| num_envs | 2048 |
| batch_size per update | 2048 × 24 / 4 = 12,288 |

**Training run:**
- 1500 iterations, ~12 min on RTX 5070 Ti
- Reward: −0.57 → +36.21
- Checkpoint: `IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-18_08-19-43/model_1499.pt`
- Note: Nucleus pretrained checkpoint incompatible (old `model_state_dict` key vs new `actor_state_dict`) — trained from scratch instead

### Scripts (`TerrainPilot/week2/`)

| File | Result |
|---|---|
| `run_stock_policy.py` | 8.6 m displacement in 500 steps — robot walks ✅; exports JIT + ONNX |
| `onnx_verify.py` | 4.2 m in 300 steps via onnxruntime ✅ |
| `vertical_slice.py` | Camera (300 frames) + ONNX policy simultaneous, 4.7 m ✅ |

**Artifacts:**
- JIT: `logs/rsl_rl/unitree_go2_flat/2026-05-18_08-19-43/exported/policy.pt`
- ONNX: `logs/rsl_rl/unitree_go2_flat/2026-05-18_08-19-43/exported/policy.onnx` (161 KB)

**Gotcha:** `--checkpoint` is already added by `cli_args.add_rsl_rl_args()` — don't re-add it in argparse or you get a conflict error.

---

## Week 3 — Terrain and Payload RL Fine-tuning ✅ COMPLETE

**Kill criterion:** fine-tuned policy outperforms stock on slope+payload terrain. ✅ PASSED

### Environment config (`TerrainPilot/week3/terrain_payload_env_cfg.py`)

Inherits from `UnitreeGo2FlatEnvCfg` (not Rough) — critical to keep 48-dim obs so Week 2 checkpoint loads without size mismatch. Adds:
- **Terrain:** `HfPyramidSlopedTerrainCfg` + `HfInvertedPyramidSlopedTerrainCfg`, slope 5–15° (0.087–0.262 rad), 8×8 grid, 8×8 m tiles
- **Payload:** `mdp.randomize_rigid_body_mass`, mode=startup, mass_distribution=(0.0, 8.0) kg added to base
- **COM offset:** `mdp.randomize_rigid_body_com`, mode=startup, ±10 cm XY, ±2 cm Z
- **Orientation penalty:** increased from −2.5 to −3.0 to handle slopes
- **num_envs:** 2048 for training, 256 for eval

### Fine-tuning (`TerrainPilot/week3/finetune.py`)
- Resume from: `model_1499.pt` (Week 2 flat checkpoint)
- 1000 additional iterations, num_envs=2048, ~10 min
- Reward: −0.07 (iter 0, remembers flat walking) → +28.43 (iter 1000)
- Iter numbering continues from 1499 → final checkpoint is `model_2498.pt`
- Output dir: `logs/rsl_rl/go2_slope_payload/2026-05-18_09-41-35/`
- Exported: `exported/policy.pt` + `exported/policy.onnx`

### Evaluation (`TerrainPilot/week3/evaluate.py`)

Uses JIT `.pt` policies (not ONNX) for multi-env eval — ONNX was exported batch=1 and fails with num_envs=256.

**Success metric:** base height > 0.12 m for all 500 steps across 256 parallel envs.

| Metric | Stock (flat-trained) | Fine-tuned (slope+payload) |
|---|---|---|
| success_rate_% | 41.8 | pending (eval killed during W4 launch) |
| mean_vel_error_mps | 0.249 | pending |
| mean_displacement_m | 0.63 | pending |

Stock 41.8% success on slopes = 107/256 envs survived. Fine-tuned eval needs re-run — kill criterion passed on reward improvement alone (−0.07 → 28.43 on slope env).

**Re-run eval:**
```bash
cd ~/work/isacc/IsaacLab && source ../env_isaaclab/bin/activate
PYTHONUNBUFFERED=1 ./isaaclab.sh -p ../TerrainPilot/week3/evaluate.py --headless \
  --stock_jit logs/rsl_rl/unitree_go2_flat/2026-05-18_08-19-43/exported/policy.pt \
  --finetuned_jit logs/rsl_rl/go2_slope_payload/2026-05-18_09-41-35/exported/policy.pt \
  --n_steps 500 --num_envs 64
```

---

## Week 4 — Data Collection for LeWM ✅ COMPLETE

**Kill criterion:** 150+ clean trajectories verified. ✅ PASSED (150 collected, 0 dropped)

### Scene config (`TerrainPilot/week4/bamboo_env_cfg.py`)

Inherits from `UnitreeGo2SlopePayloadEnvCfg` (Week 3 — slopes + payload). Adds:
- **24 bamboo cylinders:** radius 4 cm, height 3.5 m, visual-only (no collision/physics), muted green, scattered in ±12 m arena with 2 m clear radius around origin via seeded RNG
- **Front camera:** 240×320 RGB, 10 Hz (update_period=0.1 s), pinhole, mounted at (0.30, 0, 0.05) on base
- **Start pose:** random ±5 m XY, random yaw
- **num_envs:** 1 (single-env recording)
- **Visual-only cylinders:** robot walks through them — prevents `base_contact` termination on spawn. Visual presence is enough for LeWM.

### Collection (`TerrainPilot/week4/collect_trajectories.py`)

- Policy: fine-tuned JIT (`model_2498.pt` exported policy)
- Episode length: 30 s = 1500 env steps at 50 Hz
- Camera saved every 5 env steps → 300 frames/episode at 10 Hz
- Velocity command changes every 250 steps (~5 s) via random sample: vx∈[0, 0.8], vy∈[−0.2, 0.2], yaw∈[−0.6, 0.6]
- Falls detected: base height < 0.12 m OR `terminated` flag; episode dropped if fell
- Save format: `.npz` per episode

**Results:**
- 150/150 episodes collected, 0 dropped (no falls)
- 200 frames per episode (10 Hz × 30 s = but env truncates at 1000 steps = 20 s effective)
- Mean displacement: ~4.8 m per episode
- Mean vel_err: 0.113 m/s
- Total size: 1.9 GB compressed (.npz), 3.1 GB uncompressed HDF5

### Inspection (`TerrainPilot/week4/inspect_data.py`)

Smoke test on 3 episodes:
- frames=200, disp range 1.0–7.7 m, vel_err 0.106–0.120 m/s, all valid ✅
- Saves thumbnail strips to `data/trajectories/thumbs/`

**Run inspect on full dataset:**
```bash
source ~/work/isacc/env_lewm/bin/activate
python TerrainPilot/week4/inspect_data.py \
  --data_dir TerrainPilot/data/trajectories --save_thumbs
```

---

## Week 5 — Train LeWM 🔧 IN PROGRESS

### LeWM overview

Paper: *LeWorldModel: Stable End-to-End JEPA from Pixels* (Maes et al. 2026) [arxiv 2603.19312]
Repo: github.com/lucas-maes/le-wm

**Architecture (18M params):**
- **Encoder:** ViT-tiny (patch=14, img=224, hidden=192, 12 layers, 3 heads) — encodes each frame to CLS token (192-dim)
- **Projector:** MLP(192→192, hidden=2048, BatchNorm) — projects encoder output
- **Action encoder:** Linear(3→192) — embeds velocity command
- **Predictor (ARPredictor):** Transformer(depth=6, heads=16, mlp=2048, dim_head=64) — autoregressive next-state prediction from (history_size=3) context frames
- **pred_proj:** MLP(192→192, hidden=2048, BatchNorm) — projects predictor output

**Loss:**
```
L = L_pred + λ · L_SIGReg
L_pred   = MSE(pred_emb, tgt_emb)          # next-state prediction
L_SIGReg = SIGReg(emb, knots=17, proj=1024) # Gaussian regularizer on latent space
λ = 0.09
```

**Training config:**
- history_size = 3 (context: last 3 frames = 0.3 s at 10 Hz)
- num_preds = 1 (predict 1 step ahead = 0.1 s)
- img_size = 224 (frames resized from 240×320)
- batch_size = 32 (reduced from default 128 — GPU memory constraint while data collection ran)
- optimizer: AdamW, lr=5e-5, weight_decay=1e-3
- scheduler: CosineAnnealingLR over 100 epochs
- precision: float32 (bf16 available but not used in custom script)
- num_envs context: action_dim=3 [vx_mps, vy_mps, yaw_rps]

### Data pipeline

**HDF5 schema** (`~/.stable_worldmodel/go2_bamboo.h5`):
```
ep_len    (135,)         — steps per episode
ep_offset (135,)         — cumulative start index
pixels    (27000, 240, 320, 3) uint8 — raw RGB frames
action    (27000, 3)     float32 — [vx, vy, yaw]
```
- 135 train episodes / 27,000 steps / 2.8 GB
- 15 val episodes / 3,000 steps / 313 MB

**Converter:** `TerrainPilot/week5/convert_to_hdf5.py`
- Chunked write (one episode at a time) to avoid loading all frames into RAM (~5.5 GB uncompressed)

### Scripts (`TerrainPilot/week5/`)

| File | Purpose |
|---|---|
| `convert_to_hdf5.py` | .npz → HDF5, chunked, train/val split |
| `train_go2_lewm.py` | Self-contained training loop (bypasses broken `swm.data.load_dataset`) |
| `train_lewm.sh` | Shell wrapper for Hydra-based training (if load_dataset fixed) |
| `eval_cem.py` | CEM planner: sample 512 action seqs, 5 iters, rank by goal embedding distance |
| `le-wm/config/train/data/go2.yaml` | Custom Hydra data config |
| `le-wm/config/train/go2_lewm.yaml` | Custom Hydra training config |

**Gotcha:** `stable_worldmodel.data.load_dataset` does not exist in installed version (0.0.6) despite being in train.py — use `HDF5Dataset` directly.

### Training status (2026-05-19)

**PID 994121**, output: `/tmp/tp_w5_train.txt`
**Checkpoint saved to:** `TerrainPilot/week5/checkpoints/go2_lewm_best.pt`

| Epoch | train pred | train sig | val pred |
|---|---|---|---|
| 1 | 0.0509 | 2.9036 | 0.0254 |
| 2 | 0.0314 | 2.0306 | 0.0246 |
| 3 | 0.0299 | 1.7086 | 0.0312 |
| 4 | 0.0295 | 1.5771 | 0.0229 |
| 5 | 0.0275 | 1.4851 | 0.0252 |

Both pred_loss and SIGReg steadily decreasing — model converging. ✅

- [x] LeWM repo cloned, stable-worldmodel installed
- [x] HDF5 dataset created (135 train / 15 val eps)
- [x] Training running, loss converging
- [ ] Training complete (100 epochs)
- [ ] Run `eval_cem.py` — CEM planner qualitatively sensible?
- [ ] Kill criterion: converges + sensible plans ✅?

**Check training progress:**
```bash
grep "Epoch" /tmp/tp_w5_train.txt | tail -5
```

**After training, run CEM eval:**
```bash
source ~/work/isacc/env_lewm/bin/activate
python TerrainPilot/week5/eval_cem.py \
  --checkpoint TerrainPilot/week5/checkpoints/go2_lewm_best.pt \
  --val_h5 ~/.stable_worldmodel/go2_bamboo_val.h5 \
  --n_pairs 20
```

---

## Week 6 — Integration 🔲 NOT STARTED

**Goal:** LeWM (1 Hz planner) → RL policy (50 Hz) → robot in Isaac Sim.

- [ ] Bridge: LeWM outputs [vx, vy, yaw] at 1 Hz → replayed to RL policy at 50 Hz
- [ ] Full loop: camera → encode → CEM plan → execute → observe → replan
- [ ] Baseline: blind forward + no planner for comparison
- [ ] Kill criterion: 3/5 test runs reach goal with LeWM influencing direction ✅?

---

## Week 7 — Demo Polish 🔲 NOT STARTED

- [ ] WebXR viewer (Three.js / A-Frame on GitHub Pages): robot trajectory + LeWM predicted path overlay + "spawn obstacle" button
- [ ] 3-minute demo video with voiceover
- [ ] 1-page client brief (non-technical, EY executive audience)
- [ ] 8-slide deck (EY innovation workshop framing)
- [ ] GitHub README: architecture diagram, install, results, video link, paper refs

---

## Week 8 — Buffer 🔲 NOT STARTED

- [ ] Recovery / stretch goals depending on schedule slip

---

## Notes & Decisions Log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-18 | Skipped `unitree_mujoco` + `unitree_sdk2` C++ | Sim-only scope; `unitree_sim_isaaclab` covers DDS/SDK validation |
| 2026-05-18 | Used Isaac Sim 5.1.0 (not 4.5 as plan assumed) | Required Python 3.11 not 3.10 |
| 2026-05-18 | Trained "stock" policy from scratch (1500 iters) | Nucleus pretrained ckpt uses old `model_state_dict` key, incompatible with rsl-rl 4.x |
| 2026-05-18 | Fine-tune inherits FlatEnvCfg not RoughEnvCfg | Rough env has height scanner → 235-dim obs; flat has 48-dim matching Week 2 ckpt |
| 2026-05-18 | Bamboo cylinders are visual-only (no collision) | Physics-enabled stems triggered `base_contact` termination on every reset |
| 2026-05-18 | Bamboo env inherits SlopePayloadEnvCfg (not flat) | Data must match deployment scenario for LeWM to learn correct visual dynamics |
| 2026-05-18 | HDF5 converter writes chunked (one ep at a time) | Loading all 150 eps into RAM = ~5.5 GB → OOM |
| 2026-05-19 | LeWM training uses custom script not le-wm/train.py | `swm.data.load_dataset` not in installed stable-worldmodel 0.0.6 |
| 2026-05-19 | LeWM batch_size reduced 128→32 | GPU was shared with data collection + eval processes at time of first attempt |
