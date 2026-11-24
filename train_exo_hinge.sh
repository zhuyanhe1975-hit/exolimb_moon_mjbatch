#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CPU_PYTHON="${PLAYGROUND_PYTHON:-${CPU_PYTHON:-$HOME/dev-envs/robotics-playground-py312/bin/python}}"
base="$root/bests/exo_jump_force_wangyu_hut_ball_aft_v27_100m_boot18v_2026-11-23/model_399.pt"

exec "$CPU_PYTHON" "$root/scripts/train_cpu.py" \
  --task exo_hinge_jump \
  --init-checkpoint "$base" \
  --num-envs 8192 --steps-per-env 32 \
  --terrain-size 100 --geometry fit3d_18v \
  --wangyu-eva --suit-ankle-pitch --suit-major-joints --relax-propulsion --slope-adaptive-posture \
  --exo-mechanism hut_hinge_aft_v7 --exo-control impedance-hinge-force \
  --assist-force 500 --hinge-torque 100 --assist-drop-prob 0.05 \
  --stance-transition --roll-balance --long-hop-period 1.7 --stance-center-weight 22 \
  --human-energy-weight 5.5 --torque-weight 2.2 --unloaded-linear-cost 0.5 \
  --assist-share-weight 140 --speed-track-boost 90 \
  --lr 1e-4 --lr-schedule fixed --iterations 400 --seed 281 \
  --physics-backend warp --update-device cuda --torch-threads 2 "$@"
