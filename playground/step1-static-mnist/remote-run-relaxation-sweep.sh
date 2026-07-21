#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CYBERENGINE_ENTRYPOINT="playground/step1-static-mnist/relaxation_sweep.py"

if (($# == 0)); then
  set -- \
    --relaxation-steps 1 5 10 20 \
    --seeds 7 42 123 \
    --output "playground/step1-static-mnist/runs/relaxation-sweep-v1.json"
fi

exec "${script_dir}/remote-run.sh" "$@"
