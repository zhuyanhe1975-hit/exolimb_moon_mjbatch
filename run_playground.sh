#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
python_bin="${PLAYGROUND_PYTHON:-$HOME/dev-envs/robotics-playground-py312/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Playground interpreter not found: $python_bin. Set PLAYGROUND_PYTHON; see $root/README.md" >&2
  exit 1
fi
exec "$python_bin" launch.py play "$@"
