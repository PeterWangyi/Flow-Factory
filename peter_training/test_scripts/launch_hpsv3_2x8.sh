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

# Launch one node of a 2-node x 8-GPU HPSv3 GRPO run.
#
# Positional usage:
#   bash peter_training/test_scripts/launch_hpsv3_2x8.sh \
#     <config.yaml> <machine_rank> [ff-train args...]
#
# Environment-only usage:
#   FF_CONFIG=examples/grpo/lora/qwen_image/hpsv3_2x8.yaml \
#     MACHINE_RANK=0 bash peter_training/test_scripts/launch_hpsv3_2x8.sh

set -euo pipefail

ff_config="${1:-${FF_CONFIG:-}}"
if [[ $# -gt 0 ]]; then shift; fi

ff_machine_rank="${1:-${MACHINE_RANK:-${NODE_RANK:-}}}"
if [[ $# -gt 0 ]]; then shift; fi



# ff_master_addr="${1:-${MASTER_ADDR:-${MASTER_IP:-}}}"
# if [[ $# -gt 0 ]]; then shift; fi

ff_master_port="${MASTER_PORT:-29501}"
# ff_num_machines="${NUM_MACHINES:-2}"
ff_num_machines=2

# ff_master_addr=10.119.29.113 # 01
ff_master_addr=10.119.29.114 # 23
# ff_master_addr=10.119.18.14 # 45
# ff_master_addr=10.119.29.112 # 67


# cd /mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory && conda activate /mnt/aigc/wangyubo/anaconda3/envs/flowfactory


# bash peter_training/test_scripts/launch_hpsv3_2x8.sh examples/grpo/lora/z_image/reward_ab/z-image-data-u15human-reward-hpsv3.yaml 0

# bash peter_training/test_scripts/launch_hpsv3_2x8.sh examples/grpo/lora/z_image/reward_ab/z-image-data-u15human-reward-v020realism.yaml 0

# pkill -KILL -f 'peter_training/test_scripts/launch_hpsv3_2x8.sh'

# /mnt/aigc/yanglei/.conda/envs/vlm_eval_kit/bin/hf download \
#   Thunderbolt215215/UniPercept \
#   --local-dir /mnt/aigc/zoemodels/UniPercept
# /mnt/aigc/zoemodels/UniPercept

ff_gpus_per_node="${GPUS_PER_NODE:-8}"
ff_dry_run="${FF_DRY_RUN:-0}"

ff_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ff_repo_root="$(cd "${ff_script_dir}/../.." && pwd)"

if [[ -z "${ff_config}" ]]; then
  echo "config is required. Pass a YAML path as argument 1 or set FF_CONFIG." >&2
  exit 2
fi
if [[ "${ff_config}" = /* ]]; then
  ff_config_path="${ff_config}"
else
  ff_config_path="${ff_repo_root}/${ff_config}"
fi
if [[ ! -f "${ff_config_path}" ]]; then
  echo "Training config does not exist: ${ff_config_path}" >&2
  exit 1
fi
ff_config_key="$(basename "${ff_config}")"
ff_config_key="${ff_config_key%.*}"

if [[ -z "${ff_machine_rank}" ]]; then
  echo "machine_rank is required. Pass it as argument 2 or set MACHINE_RANK." >&2
  exit 2
fi
if [[ "${ff_machine_rank}" != "0" && "${ff_machine_rank}" != "1" ]]; then
  echo "machine_rank must be 0 or 1 for this 2-node launcher; got '${ff_machine_rank}'." >&2
  exit 2
fi
if [[ -z "${ff_master_addr}" ]]; then
  echo "ff_master_addr must be configured for this launcher." >&2
  exit 2
fi
if [[ "${ff_num_machines}" != "2" ]]; then
  echo "NUM_MACHINES must be 2 for this launcher; got '${ff_num_machines}'." >&2
  exit 2
fi
if [[ "${ff_gpus_per_node}" != "8" ]]; then
  echo "GPUS_PER_NODE must be 8 for this launcher; got '${ff_gpus_per_node}'." >&2
  exit 2
fi

export MASTER_ADDR="${ff_master_addr}"
export MASTER_PORT="${ff_master_port}"
export NUM_MACHINES="${ff_num_machines}"
export GPUS_PER_NODE="${ff_gpus_per_node}"
export MACHINE_RANK="${ff_machine_rank}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export WANDB_MODE="${WANDB_MODE:-online}"

echo "=== Flow-Factory HPSv3 2x8 launch ==="
echo "Config:         ${ff_config}"
echo "Master:         ${MASTER_ADDR}:${MASTER_PORT}"
echo "Machine rank:   ${MACHINE_RANK}"
echo "Topology:       ${NUM_MACHINES} nodes x ${GPUS_PER_NODE} GPUs"

cd "${ff_repo_root}"

if [[ "${ff_dry_run}" == "1" ]]; then
  printf 'Dry-run command: '
  printf '%q ' ff-train "${ff_config}" "$@"
  printf '\n'
  exit 0
fi

if ! command -v ff-train >/dev/null 2>&1; then
  echo "ff-train is not available in PATH. Activate the Flow-Factory environment." >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1; then
  echo "python is not available in PATH. Activate the Flow-Factory environment." >&2
  exit 1
fi

ff_gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${ff_gpu_count}" != "${GPUS_PER_NODE}" ]]; then
  echo "Expected ${GPUS_PER_NODE} visible GPUs, but PyTorch reports ${ff_gpu_count}." >&2
  echo "Check CUDA_VISIBLE_DEVICES and the active PyTorch environment." >&2
  exit 1
fi

ff_log_dir="${FF_LOG_DIR:-${ff_repo_root}/logs/hpsv3_2x8}"
mkdir -p "${ff_log_dir}"
ff_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
ff_log_file="${ff_log_dir}/${ff_config_key}_node${MACHINE_RANK}_${ff_timestamp}.log"

echo "Log:            ${ff_log_file}"
echo

unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE

ff-train "${ff_config}" "$@" 2>&1 | tee "${ff_log_file}"
