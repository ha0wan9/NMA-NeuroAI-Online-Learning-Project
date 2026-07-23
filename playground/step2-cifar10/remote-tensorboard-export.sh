#!/usr/bin/env bash
set -euo pipefail

remote_host="${CYBERENGINE_HOST:-cyberengine}"
remote_root="${CYBERENGINE_WORKSPACE:-/home/haoran/.cache/nma-neuroai-step2}"
runtime_image="${CYBERENGINE_IMAGE:-localhost/nma-neuroai-step1-rocm:7.2.1}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
run_dir="playground/step2-cifar10/runs"

if (($# == 0)); then
  set -- "${run_dir}/split-cifar10-v1.json"
fi

ssh "${remote_host}" mkdir -p "${remote_root}/.python-packages" "${remote_root}/${run_dir}"
rsync --archive \
  "${script_dir}/tensorboard_export.py" \
  "${script_dir}/representation_viz.py" \
  "${remote_host}:${remote_root}/playground/step2-cifar10/"

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

printf -v quoted_artifacts ' %q' "$@"
ssh "${remote_host}" podman run --rm \
  --security-opt=label=disable \
  --volume="${remote_root}:/workspace" \
  --workdir=/workspace \
  --env=PYTHONPATH=/workspace/.python-packages \
  "${runtime_image}" \
  python3 playground/step2-cifar10/tensorboard_export.py${quoted_artifacts}

mkdir -p "${repo_root}/${run_dir}/tensorboard"
rsync --archive \
  "${remote_host}:${remote_root}/${run_dir}/tensorboard/" \
  "${repo_root}/${run_dir}/tensorboard/"
