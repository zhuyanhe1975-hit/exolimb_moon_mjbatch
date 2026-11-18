#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$root/run_cpu.sh" \
  --checkpoint "$root/logs/2026-11-15_import_mjlab_v15_groundsync1/model_200.pt" "$@"
