#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
python_bin="${CPU_PYTHON:-$root/.venv-cpu/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "CPU Python not found: $python_bin. See README.md for .venv-cpu setup." >&2
  exit 1
fi
set -- --latest-training "$@"
use_window=true
previous=""
for arg in "$@"; do
  if [[ "$previous" == --viewer && "$arg" == studio ]]; then use_window=false; fi
  case "$arg" in --viewer=studio|--headless|--help|-h) use_window=false ;; esac
  previous="$arg"
done
if [[ "$(uname -s)" == Darwin && "$use_window" == true ]]; then
  exec "$python_bin" "$(dirname "$python_bin")/mjpython" scripts/play_cpu.py "$@"
fi
exec "$python_bin" scripts/play_cpu.py "$@"
