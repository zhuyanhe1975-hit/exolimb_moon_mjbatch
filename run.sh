#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
backend=cpu
backend_seen=false
args=()
while (( $# )); do
  case "$1" in
    --backend|--backend=*)
      if [[ "$backend_seen" == true ]]; then
        echo "Specify --backend only once." >&2
        exit 2
      fi
      backend_seen=true
      if [[ "$1" == --backend ]]; then
        if (( $# < 2 )); then
          echo "--backend requires cpu or playground." >&2
          exit 2
        fi
        shift
        backend="$1"
      else
        backend="${1#*=}"
      fi
      ;;
    *) args+=("$1") ;;
  esac
  shift
done
case "$backend" in
  cpu) ;;
  playground)
    for arg in "${args[@]}"; do
      case "$arg" in
        --best|--best=*|--list-bests|--latest-training|--task|--task=*|--jump-*|--assist-force|--assist-force=*)
          echo "$arg is a CPU replay option. Use ./run.sh --backend cpu." >&2
          exit 2
          ;;
      esac
    done
    exec "$root/run_playground.sh" "${args[@]}"
    ;;
  *)
    echo "Unknown backend: $backend. Choose cpu or playground." >&2
    exit 2
    ;;
esac
export MOON_RUN_ENTRY=run.sh
exec "$root/run_cpu.sh" "${args[@]}"
