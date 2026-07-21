#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CYBERENGINE_ENTRYPOINT="playground/step2-cifar10/split_experiment.py"

if (($# == 0)); then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  set -- \
    --seeds 42 \
    --epochs-per-task 1 \
    --max-train-samples-per-task 40 \
    --max-eval-samples-per-task 40 \
    --output "playground/step2-cifar10/runs/split-smoke-${timestamp}.json"
fi

exec "${script_dir}/remote-run.sh" "$@"
