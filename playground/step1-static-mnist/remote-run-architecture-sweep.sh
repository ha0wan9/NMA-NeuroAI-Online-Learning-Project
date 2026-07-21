#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CYBERENGINE_ENTRYPOINT="playground/step1-static-mnist/architecture_sweep.py"

if (($# == 0)); then
  set -- \
    --seeds 7 42 123 \
    --pc-steps 5 \
    --output "playground/step1-static-mnist/runs/architecture-sweep-v2.json"
fi

exec "${script_dir}/remote-run.sh" "$@"
