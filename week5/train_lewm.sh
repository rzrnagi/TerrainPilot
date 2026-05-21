#!/usr/bin/env bash
# Day 3-5: Train LeWM on Go2 bamboo-forest trajectories.
#
# Prerequisites:
#   1. python week5/convert_to_hdf5.py --data_dir data/trajectories
#   2. WandB login: source env_lewm/bin/activate && wandb login
#
# Run from ~/work/isacc/:
#   bash TerrainPilot/week5/train_lewm.sh

set -e
cd "$(dirname "$0")/le-wm"

source ../../env_lewm/bin/activate

echo "[TerrainPilot W5] Starting LeWM training on Go2 data..."
echo "  Config:   go2_lewm"
echo "  Data:     ~/.stable_worldmodel/go2_bamboo.h5"
echo "  Device:   GPU (bf16)"
echo ""

python train.py \
  --config-name=go2_lewm \
  trainer.max_epochs=100 \
  loader.batch_size=64 \
  "hydra.run.dir=outputs/go2_lewm/\$(date +%Y-%m-%d_%H-%M-%S)"
