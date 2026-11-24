#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 0 && ("$1" == --help || "$1" == -h) ]]; then
  echo "Usage: $0 [CHECKPOINT] [replay options]"
  echo "Defaults: exo_hinge_jump, Studio, skin, 100m terrain, 1.5 m/s, 10-second episodes."
  echo "Omit CHECKPOINT to load the latest run under logs/cpu_exo_hinge_train/."
  echo "Use --hinge-torque 0 or --assist-force 0 for ablation comparisons."
  exit 0
fi

ckpt_args=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  checkpoint="$1"
  shift
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
  fi
  checkpoint="$(cd -- "$(dirname -- "$checkpoint")" && pwd)/$(basename -- "$checkpoint")"
  ckpt_args=(--checkpoint "$checkpoint")
fi

exec "$root/run_cpu.sh" --task exo_hinge_jump --viewer studio --viz skin \
  --terrain-size 100 --jump-speed 1.5 --episode-seconds 30 \
  "${ckpt_args[@]}" "$@" --require-force-control
