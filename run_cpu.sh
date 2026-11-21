#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
python_bin="${CPU_PYTHON:-$root/.venv-cpu/bin/python}"
platform="$(uname -s)"
if [[ "$platform" == Linux && -z "${CPU_PYTHON:-}" && ! -x "$python_bin" ]]; then
  shared_python="$HOME/dev-envs/robotics-playground-py312/bin/python"
  if [[ -x "$shared_python" ]]; then python_bin="$shared_python"; fi
fi
if [[ ! -x "$python_bin" ]]; then
  for arg in "$@"; do
    case "$arg" in
      --help|-h|--list-bests) python_bin="$(command -v python3 || true)"; break ;;
    esac
  done
fi
if [[ ! -x "$python_bin" ]]; then
  echo "CPU Python not found: $python_bin. See README.md for .venv-cpu setup." >&2
  exit 1
fi
export MOON_RUN_ENTRY="${MOON_RUN_ENTRY:-run_cpu.sh}"
use_window=true
previous=""
for arg in "$@"; do
  if [[ "$previous" == --viewer && "$arg" == studio ]]; then use_window=false; fi
  case "$arg" in --viewer=studio|--headless|--help|-h|--list-bests) use_window=false ;; esac
  previous="$arg"
done
if [[ "$platform" == Darwin && "$use_window" == true ]]; then
  exec "$python_bin" "$(dirname "$python_bin")/mjpython" scripts/play_cpu.py "$@"
fi
exec "$python_bin" scripts/play_cpu.py "$@"
