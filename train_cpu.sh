#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
python_bin="${CPU_PYTHON:-$root/.venv-cpu/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "CPU Python not found: $python_bin. See README.md for setup." >&2
  exit 1
fi
exec "$python_bin" scripts/train_cpu.py "$@"
