#!/usr/bin/env bash
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

### common
WORKSPACE_NAME="aigc"
PRIORITY="HIGHEST"
TRAINING_FRAMEWORK="pytorch"

### container image
CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/ccr_2/ulimit-change:20240725-18h11m01s"

### aec2 name
# AEC2_NAME="umm"
# AEC2_NAME="m-train-neo2-interleave"
# AEC2_NAME="m-train-neo2"
# AEC2_NAME="vigen"
AEC2_NAME="m-train-neo2-infographic"
# AEC2_NAME="neo1-edit"
# AEC2_NAME="neo1-trajectory"
# AEC2_NAME="neo1-agentic"
# AEC2_NAME="si"
# AEC2_NAME="umm"

### storage mount
STORAGE_MOUNT="1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs,c83d08bc-2965-11ef-b8c5-929f74fd8884:/mnt/aigc/,01998fb1-b876-7b33-82c9-4427517bf536:/mnt/umm"

### worker spec
WORKER_NODES=2
WORKER_SPEC="N6lS.Iu.I80.8"  # default
# WORKER_SPEC="N6lS.Iq.I10.8"  # vigen si

GPUS_PER_NODE=8

### training command
REPO_ROOT="/mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory"
CONDA_SH="/mnt/aigc/wangyubo/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="/mnt/aigc/wangyubo/anaconda3/envs/flowfactory"

LAUNCH_SCRIPT="${REPO_ROOT}/peter_training/test_scripts/launch_hpsv3_2x8.sh"
CONFIG_FILE="${REPO_ROOT}/examples/grpo/lora/z_image/reward_ab/z-image-data-u15human-reward-v020realism.yaml"
JOB_NAME="FF-ZImage-u15human-v020realism"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
  echo "Launcher not found: ${LAUNCH_SCRIPT}" >&2
  exit 1
fi

MASTER_PORT=29500
WANDB_MODE="online"
WANDB_API_KEY="${WANDB_API_KEY:-}"

COMMAND=$(cat <<EOF
bash -lc '
set -euo pipefail

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

export NUM_MACHINES="\${WORLD_SIZE:-${WORKER_NODES}}"
export GPUS_PER_NODE="${GPUS_PER_NODE}"
export MACHINE_RANK="\${RANK:-0}"
export MASTER_ADDR="\${MASTER_ADDR:?MASTER_ADDR_is_required_from_cluster}"
export MASTER_PORT="${MASTER_PORT}"

export WANDB_MODE="${WANDB_MODE}"
export WANDB_API_KEY="${WANDB_API_KEY}"

bash "${LAUNCH_SCRIPT}" "${CONFIG_FILE}" "\${MACHINE_RANK}"
'
EOF
)

sco acp jobs create \
    --workspace-name="${WORKSPACE_NAME}" \
    --aec2-name="${AEC2_NAME}" \
    --job-name="${JOB_NAME}" \
    --priority="${PRIORITY}" \
    --container-image-url="${CONTAINER_IMAGE_URL}" \
    --storage-mount="${STORAGE_MOUNT}" \
    --training-framework="${TRAINING_FRAMEWORK}" \
    --worker-nodes="${WORKER_NODES}" \
    --worker-spec="${WORKER_SPEC}" \
    --command="${COMMAND}"
