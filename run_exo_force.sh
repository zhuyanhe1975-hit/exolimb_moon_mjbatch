#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 || "$1" == --help || "$1" == -h ]]; then
  echo "Usage: $0 CHECKPOINT [replay options]"
  echo "Defaults: Studio, skin, 1.0 m/s, 10-second episodes, and the checkpoint's assist limit."
  echo "Use --assist-force 0 for the unpowered comparison."
  if [[ $# -eq 0 ]]; then exit 2; fi
  exit 0
fi

checkpoint="$1"
shift
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 2
fi
checkpoint="$(cd -- "$(dirname -- "$checkpoint")" && pwd)/$(basename -- "$checkpoint")"

exec "$root/run_exo_jump.sh" --checkpoint "$checkpoint" --episode-seconds 10 "$@" --require-force-control
