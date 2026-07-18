#!/usr/bin/env bash
set -euo pipefail

remote_host="${CYBERENGINE_HOST:-cyberengine}"
runtime_image="${CYBERENGINE_IMAGE:-localhost/nma-neuroai-step1-rocm:7.2.1}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remote_setup_dir="${CYBERENGINE_SETUP_DIR:-/home/haoran/.cache/nma-neuroai-step1-setup}"

ssh "${remote_host}" mkdir -p "${remote_setup_dir}"
rsync --archive "${script_dir}/Containerfile.cyberengine" "${remote_host}:${remote_setup_dir}/"
ssh "${remote_host}" \
  "podman build --tag ${runtime_image} --file ${remote_setup_dir}/Containerfile.cyberengine ${remote_setup_dir}"

ssh "${remote_host}" "podman run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=keep-groups \
  --security-opt=label=disable \
  ${runtime_image} \
  python3 -c 'import torch, torchvision, pandas, matplotlib, seaborn, tqdm; \
print(\"torch\", torch.__version__); \
print(\"hip\", torch.version.hip); \
print(\"gpu_available\", torch.cuda.is_available()); \
print(\"device\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); \
assert torch.cuda.is_available(); \
x=torch.randn(1024,1024,device=\"cuda\"); y=x@x; torch.cuda.synchronize(); \
print(\"matmul\", tuple(y.shape), y.device)'"
