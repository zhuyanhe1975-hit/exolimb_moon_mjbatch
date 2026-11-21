#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CPU_PYTHON="${PLAYGROUND_PYTHON:-${CPU_PYTHON:-$HOME/dev-envs/robotics-playground-py312/bin/python}}"
base="$root/logs/cpu_exo_train/stance_center_repro_20261119_084445/stage2/model_239.pt"

exec "$root/train_exo_jump.sh" \
  --init-checkpoint "$base" --num-envs 8192 --steps-per-env 32 \
  --terrain-idxs 12 3 24 47 --geometry fitted --assist-force 500 \
  --stance-center-weight 22 --iterations 160 --seed 42 \
  --physics-backend warp --update-device cuda --torch-threads 2 \
  --exo-control force-action "$@"
