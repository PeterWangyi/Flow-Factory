#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FLOW_GRPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

REWARD_MODEL_TAG=""
REWARD_MODEL_NAME=""
MODEL_REGISTRY_PATH="${MODEL_REGISTRY_PATH:-${SCRIPT_DIR}/models.json}"
REGISTRY_PYTHON="${REGISTRY_PYTHON:-}"
DRY_RUN=0
LIST_TAGS=0

usage() {
  cat <<EOF
Usage:
  bash ${BASH_SOURCE[0]} --tag TAG [--model MODEL] [options]
  bash ${BASH_SOURCE[0]} --list-tags

Options:
  --tag TAG          Tag or tag prefix, for example 021 or 021_realism
  --model MODEL      Optional model name appended as TAG_MODEL
  --nodes N          Number of one-GPU worker nodes
  --registry PATH    Override models.json
  --dry-run          Resolve and validate without submitting
  --list-tags        Validate and list registered tags
  -h, --help         Show this help
EOF
}

require_option_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "${option} requires a value." >&2
    usage >&2
    exit 2
  fi
}

REQUESTED_NODES=""
while (( $# > 0 )); do
  case "$1" in
    --tag)
      require_option_value "$1" "${2:-}"
      REWARD_MODEL_TAG="$2"
      shift 2
      ;;
    --model)
      require_option_value "$1" "${2:-}"
      REWARD_MODEL_NAME="$2"
      shift 2
      ;;
    --nodes)
      require_option_value "$1" "${2:-}"
      REQUESTED_NODES="$2"
      shift 2
      ;;
    --registry)
      require_option_value "$1" "${2:-}"
      MODEL_REGISTRY_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --list-tags)
      LIST_TAGS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

### common
WORKSPACE_NAME="aigc"
PRIORITY="HIGHEST"
TRAINING_FRAMEWORK="pytorch"

# container image
CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/ccr_2/ulimit-change:20240725-18h11m01s"
# CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/ccr_2/wan21-v4:20250325-16h55m11s"      # qwen ocr 
# CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/ccr_2/pytorch:2.5.1-cuda12.4-cudnn9-devel"
# CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/ccr_2/fvg_base_25:20250427_43b214e"  # with vllm

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

### job name
JOB_NAME="U15-AethModel"

### storage mount
STORAGE_MOUNT="1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs,c83d08bc-2965-11ef-b8c5-929f74fd8884:/mnt/aigc/,01998fb1-b876-7b33-82c9-4427517bf536:/mnt/umm"

### worker spec
WORKER_NODES=1
WORKER_SPEC="N6lS.Iu.I80.1"  # default
# WORKER_SPEC="N6lS.Iq.I10.1"  # vigen si

# bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh --tag 021 --model realism

# bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh --tag 022 --model overall

# bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh --tag 022 --model color

# bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh --tag 022 --model composition

# bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh --tag 022 --model creativity

# bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh --tag 022 --model lighting

# bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh --tag 022 --model mood

# bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh --tag 022 --model textCorrectness


HPSV3_REPO="${HPSV3_REPO:-/mnt/aigc/wangyubo/code/IG/neo/RL/HPSv3}"
HPSV3_ENV="${HPSV3_ENV:-/mnt/aigc/wangyubo/anaconda3/envs/hpsv3}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
SCHEDULER_PORT="${SCHEDULER_PORT:-9010}"

if [[ -n "${REQUESTED_NODES}" ]]; then
  WORKER_NODES="${REQUESTED_NODES}"
fi
if [[ ! "${WORKER_NODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--nodes/WORKER_NODES must be a positive integer." >&2
  exit 2
fi

find_registry_python() {
  local candidates=()
  local candidate=""
  if [[ -n "${REGISTRY_PYTHON}" ]]; then
    candidates+=("${REGISTRY_PYTHON}")
  fi
  candidates+=("${HPSV3_ENV}/bin/python")
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  echo "Could not find Python for reading ${MODEL_REGISTRY_PATH}." >&2
  return 1
}

REGISTRY_PYTHON="$(find_registry_python)"

if [[ "${LIST_TAGS}" == "1" ]]; then
  exec "${REGISTRY_PYTHON}" "${SCRIPT_DIR}/model_registry.py" \
    --registry "${MODEL_REGISTRY_PATH}" list
fi

if [[ -z "${REWARD_MODEL_TAG}" ]]; then
  echo "--tag is required." >&2
  usage >&2
  exit 2
fi

MODEL_LOOKUP_TAG="${REWARD_MODEL_TAG}"
if [[ -n "${REWARD_MODEL_NAME}" ]]; then
  MODEL_LOOKUP_TAG="${REWARD_MODEL_TAG}_${REWARD_MODEL_NAME}"
fi

assignments=""
if ! assignments=$("${REGISTRY_PYTHON}" "${SCRIPT_DIR}/model_registry.py" \
    --registry "${MODEL_REGISTRY_PATH}" resolve \
    --tag "${MODEL_LOOKUP_TAG}" \
    --format shell); then
  exit 1
fi
eval "${assignments}"

if ! ZHIQIAN_INSTRUCTION_NAME=$(PYTHONPATH="${HPSV3_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${REGISTRY_PYTHON}" "${SCRIPT_DIR}/prompt_registry.py" \
    --tag "${RESOLVED_MODEL_TAG}"); then
  exit 1
fi

JOB_NAME="${JOB_NAME}-${RESOLVED_MODEL_TAG}"

printf -v flow_root_q '%q' "${FLOW_GRPO_ROOT}"
printf -v hpsv3_repo_q '%q' "${HPSV3_REPO}"
printf -v hpsv3_env_q '%q' "${HPSV3_ENV}"
printf -v model_tag_q '%q' "${RESOLVED_MODEL_TAG}"
printf -v model_path_q '%q' "${REGISTERED_CHECKPOINT_PATH}"
printf -v backend_port_q '%q' "${BACKEND_PORT}"
printf -v scheduler_port_q '%q' "${SCHEDULER_PORT}"

COMMAND="cd ${flow_root_q} && \
HPSV3_REPO=${hpsv3_repo_q} \
HPSV3_ENV=${hpsv3_env_q} \
ZHIQIAN_MODEL_TAG=${model_tag_q} \
ZHIQIAN_MODEL_PATH=${model_path_q} \
BACKEND_PORT=${backend_port_q} \
SCHEDULER_PORT=${scheduler_port_q} \
bash server/zhiqian_aeth_model/start_multinode_1gpu.sh
"

echo "Resolved Zhiqian reward service"
echo "  Tag        : ${RESOLVED_MODEL_TAG}"
echo "  Model      : ${REGISTERED_CHECKPOINT_PATH}"
echo "  Instruction: ${ZHIQIAN_INSTRUCTION_NAME}"
echo "  Nodes      : ${WORKER_NODES} x 1 GPU"
echo "  Worker spec: ${WORKER_SPEC}"
echo "  Job name   : ${JOB_NAME}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run; job was not submitted."
  echo "Command:"
  echo "${COMMAND}"
  exit 0
fi

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
