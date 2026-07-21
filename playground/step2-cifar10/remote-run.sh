#!/usr/bin/env bash
set -euo pipefail

remote_host="${CYBERENGINE_HOST:-cyberengine}"
remote_root="${CYBERENGINE_WORKSPACE:-/home/haoran/.cache/nma-neuroai-step2}"
runtime_image="${CYBERENGINE_IMAGE:-localhost/nma-neuroai-step1-rocm:7.2.1}"
entrypoint="${CYBERENGINE_ENTRYPOINT:-playground/step2-cifar10/paper_experiment.py}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
run_dir="playground/step2-cifar10/runs"
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
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  set -- \
    --seeds 1482555873 \
    --epochs 1 \
    --train-per-class 40 \
    --test-per-class 20 \
    --output "${run_dir}/paper-smoke-${timestamp}.json"
fi

has_device=false
has_tensorboard_dir=false
for argument in "$@"; do
  if [[ "${argument}" == "--device" ]]; then
    has_device=true
  fi
  if [[ "${argument}" == "--tensorboard-dir" ]]; then
    has_tensorboard_dir=true
  fi
done
if [[ "${has_device}" == false ]]; then
  set -- "$@" --device cuda
fi
if [[ "${has_tensorboard_dir}" == false ]]; then
  set -- "$@" --tensorboard-dir "${run_dir}/tensorboard"
fi

ssh "${remote_host}" mkdir -p "${remote_root}/${run_dir}" "${remote_root}/.python-packages"
rsync --archive --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.ipynb_checkpoints' \
  --exclude='data' \
  --exclude='.python-packages' \
  --exclude="${run_dir}" \
  "${repo_root}/" "${remote_host}:${remote_root}/"

runtime_image_id="$(ssh "${remote_host}" "podman image inspect --format '{{.Id}}' ${runtime_image}")"
if ! ssh "${remote_host}" test -d "${remote_root}/.python-packages/tensorboard"; then
  ssh "${remote_host}" podman run --rm \
    --security-opt=label=disable \
    --volume="${remote_root}:/workspace" \
    "${runtime_image}" \
    python3 -m pip install --disable-pip-version-check --no-input \
      --target /workspace/.python-packages --no-deps \
      tensorboard==2.20.0 tensorboard-data-server==0.7.2 \
      absl-py==2.5.0 grpcio==1.82.1 markdown==3.10.2 \
      markupsafe==3.0.3 packaging==26.2 protobuf==7.35.1 \
      setuptools==80.9.0 typing-extensions==4.16.0 werkzeug==3.1.8
fi

if ! ssh "${remote_host}" test -d "${remote_root}/.python-packages/plotly"; then
  ssh "${remote_host}" podman run --rm \
    --security-opt=label=disable \
    --volume="${remote_root}:/workspace" \
    "${runtime_image}" \
    python3 -m pip install --disable-pip-version-check --no-input \
      --target /workspace/.python-packages --no-deps \
      plotly==6.3.0 narwhals==2.5.0
fi
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
  --env=PYTHONPATH=/workspace/.python-packages \
  --volume=${remote_root}:/workspace \
  --workdir=/workspace \
  ${runtime_image} \
  python3 ${entrypoint}${quoted_args}"

set +e
ssh "${remote_host}" "${remote_command}"
run_status=$?
set -e
mkdir -p "${repo_root}/${run_dir}"
rsync --archive "${remote_host}:${remote_root}/${run_dir}/" "${repo_root}/${run_dir}/"
exit "${run_status}"
