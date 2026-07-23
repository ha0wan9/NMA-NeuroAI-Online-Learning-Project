#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CYBERENGINE_ENTRYPOINT="playground/step1-static-mnist/continual_experiment.py"

if (($# == 0)); then
  run_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  set -- \
    --scenarios split permuted \
    --seeds 42 \
    --epochs-per-task 1 \
    --batch-size 64 \
    --pc-steps 4 \
    --max-train-samples-per-task 256 \
    --max-eval-samples-per-task 256 \
    --output "playground/step1-static-mnist/runs/continual-smoke-${run_timestamp}.json"
fi

exec "${script_dir}/remote-run.sh" "$@"
