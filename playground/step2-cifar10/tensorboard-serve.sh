#!/usr/bin/env bash
set -euo pipefail

remote_host="${CYBERENGINE_HOST:-cyberengine}"
remote_root="${CYBERENGINE_WORKSPACE:-/home/haoran/.cache/nma-neuroai-step2}"
runtime_image="${CYBERENGINE_IMAGE:-localhost/nma-neuroai-step1-rocm:7.2.1}"
port="${TENSORBOARD_PORT:-6006}"

echo "TensorBoard: http://127.0.0.1:${port}"
exec ssh -L "${port}:127.0.0.1:${port}" "${remote_host}" \
  podman run --rm \
    --network=host \
    --security-opt=label=disable \
    --volume="${remote_root}:/workspace" \
    --env=PYTHONPATH=/workspace/.python-packages \
    "${runtime_image}" \
    python3 -m tensorboard.main \
      --logdir /workspace/playground/step2-cifar10/runs/tensorboard \
      --host 127.0.0.1 \
      --port "${port}"
