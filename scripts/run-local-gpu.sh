#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
  echo "usage: $0 EXPERIMENT.py [ARG ...]" >&2
  exit 2
fi

entrypoint="$1"
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
pc_submodule_root="${repo_root}/playground/predictive-coding"

if [[ ! -f "${repo_root}/${entrypoint}" ]]; then
  echo "experiment entrypoint not found: ${entrypoint}" >&2
  exit 2
fi

export EXPERIMENT_EXECUTION_HOST="$(hostname)"
export EXPERIMENT_REPOSITORY_COMMIT="$(git -C "${repo_root}" rev-parse HEAD)"
export EXPERIMENT_REPOSITORY_DIRTY=false
if [[ -n "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]]; then
  export EXPERIMENT_REPOSITORY_DIRTY=true
fi
export EXPERIMENT_PC_SUBMODULE_COMMIT="$(git -C "${pc_submodule_root}" rev-parse HEAD)"
export EXPERIMENT_PC_SUBMODULE_DIRTY=false
if [[ -n "$(git -C "${pc_submodule_root}" status --porcelain --untracked-files=normal)" ]]; then
  export EXPERIMENT_PC_SUBMODULE_DIRTY=true
fi

has_device=false
for argument in "$@"; do
  if [[ "${argument}" == "--device" ]]; then
    has_device=true
    break
  fi
done
if [[ "${has_device}" == false ]]; then
  set -- "$@" --device cuda
fi

cd "${repo_root}"
uv run --extra cuda python -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
exec uv run --extra cuda python "${entrypoint}" "$@"
