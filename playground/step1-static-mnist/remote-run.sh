#!/usr/bin/env bash
set -euo pipefail

remote_host="${CYBERENGINE_HOST:-cyberengine}"
remote_root="${CYBERENGINE_WORKSPACE:-/home/haoran/.cache/nma-neuroai-step1}"
runtime_image="${CYBERENGINE_IMAGE:-localhost/nma-neuroai-step1-rocm:7.2.1}"
experiment_entrypoint="${CYBERENGINE_ENTRYPOINT:-playground/step1-static-mnist/experiment.py}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
run_dir="playground/step1-static-mnist/runs"
pc_submodule_root="${repo_root}/playground/predictive-coding"

repository_commit="$(git -C "${repo_root}" rev-parse HEAD)"
repository_dirty=false
if [[ -n "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]]; then
  repository_dirty=true
fi
pc_submodule_commit="$(git -C "${pc_submodule_root}" rev-parse HEAD)"
pc_submodule_dirty=false
if [[ -n "$(git -C "${pc_submodule_root}" status --porcelain --untracked-files=normal)" ]]; then
  pc_submodule_dirty=true
fi

if (($# == 0)); then
  smoke_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  set -- \
    --seeds 42 \
    --epochs 1 \
    --batch-size 64 \
    --pc-steps 4 \
    --max-train-samples 256 \
    --max-valid-samples 256 \
    --max-test-samples 256 \
    --output "${run_dir}/smoke-seed42-${smoke_timestamp}.json"
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

ssh "${remote_host}" mkdir -p "${remote_root}/${run_dir}"
rsync --archive --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.ipynb_checkpoints' \
  --exclude='data' \
  --exclude="${run_dir}" \
  "${repo_root}/" "${remote_host}:${remote_root}/"

runtime_image_id="$(ssh "${remote_host}" "podman image inspect --format '{{.Id}}' ${runtime_image}")"

printf -v quoted_args ' %q' "$@"
remote_command="podman run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=keep-groups \
  --security-opt=label=disable \
  --ipc=host \
  --env=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  --env=EXPERIMENT_EXECUTION_HOST=${remote_host} \
  --env=EXPERIMENT_RUNTIME_IMAGE=${runtime_image} \
  --env=EXPERIMENT_RUNTIME_IMAGE_ID=${runtime_image_id} \
  --env=EXPERIMENT_REPOSITORY_COMMIT=${repository_commit} \
  --env=EXPERIMENT_REPOSITORY_DIRTY=${repository_dirty} \
  --env=EXPERIMENT_PC_SUBMODULE_COMMIT=${pc_submodule_commit} \
  --env=EXPERIMENT_PC_SUBMODULE_DIRTY=${pc_submodule_dirty} \
  --volume=${remote_root}:/workspace \
  --workdir=/workspace \
  ${runtime_image} \
  python3 ${experiment_entrypoint}${quoted_args}"

set +e
ssh "${remote_host}" "${remote_command}"
run_status=$?
set -e

mkdir -p "${repo_root}/${run_dir}"
rsync --archive "${remote_host}:${remote_root}/${run_dir}/" "${repo_root}/${run_dir}/"
exit "${run_status}"
