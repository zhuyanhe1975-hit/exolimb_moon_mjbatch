#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

export CPU_PYTHON="${PLAYGROUND_PYTHON:-${CPU_PYTHON:-$HOME/dev-envs/robotics-playground-py312/bin/python}}"

if (( $# > 1 )); then
  echo "Usage: $0 [new-run-directory]" >&2
  exit 2
fi

run_dir="${1:-logs/cpu_exo_train/stance_center_repro_$(date +%Y%m%d_%H%M%S)}"
base="bests/exo_jump_position_pd500_2026-11-19/model_199.pt"
common=(--num-envs 8192 --steps-per-env 32 --terrain-idxs 12 3 24 47
        --physics-backend warp --torch-threads 2
        --update-device cuda --geometry fitted --assist-force 500 --pushes)

./train_exo_jump.sh "${common[@]}" \
  --init-checkpoint "$base" --stance-center-weight 14 --iterations 80 --seed 42 \
  --log-dir "$run_dir/stage1"

./train_exo_jump.sh "${common[@]}" \
  --init-checkpoint "$run_dir/stage1/model_79.pt" --stance-center-weight 22 --iterations 240 --seed 43 \
  --log-dir "$run_dir/stage2"

printf 'Checkpoint at the same iteration as the original selection: %s\n' "$run_dir/stage2/model_50.pt"
