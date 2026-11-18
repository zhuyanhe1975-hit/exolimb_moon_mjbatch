#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
  case "$arg" in
    --viewer|--viewer=*)
      exec "$root/run_cpu.sh" --task exo_jump --viz skin --jump-speed 1.0 "$@"
      ;;
  esac
done
exec "$root/run_cpu.sh" --task exo_jump --viewer studio --viz skin --jump-speed 1.0 "$@"
