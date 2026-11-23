#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CPU_PYTHON:-$root/.venv-cpu/bin/python}"
if [[ "${1:-}" == train ]]; then
  python_bin="${PLAYGROUND_PYTHON:-${CPU_PYTHON:-$HOME/dev-envs/robotics-playground-py312/bin/python}}"
fi
if [[ ! -x "$python_bin" && -z "${CPU_PYTHON:-}" && "${1:-}" != train ]]; then
  python_bin="$HOME/dev-envs/robotics-playground-py312/bin/python"
fi
if [[ "$(uname -s)" == Darwin && "${1:-}" == play && " $* " != *" --headless "* ]]; then
  exec "$(dirname "$python_bin")/mjpython" "$root/scripts/replant.py" "$@"
fi
exec "$python_bin" "$root/scripts/replant.py" "$@"
