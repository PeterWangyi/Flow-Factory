#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FLOW_GRPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL_REGISTRY_PATH="${MODEL_REGISTRY_PATH:-${SCRIPT_DIR}/models.json}"
ZHIQIAN_MODEL_TAG="${ZHIQIAN_MODEL_TAG:-}"
ZHIQIAN_MODEL_PATH="${ZHIQIAN_MODEL_PATH:-}"

HPSV3_REPO="${HPSV3_REPO:-/mnt/aigc/wangyubo/code/IG/neo/RL/HPSv3}"
HPSV3_ENV="${HPSV3_ENV:-/mnt/aigc/wangyubo/anaconda3/envs/hpsv3}"
HPSV3_DEVICE="${HPSV3_DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
GPUS_CSV="${GPUS_CSV:-0}"
PORTS_CSV="${PORTS_CSV:-}"

START_SCHEDULER="${START_SCHEDULER:-0}"
SCHEDULER_HOST="${SCHEDULER_HOST:-0.0.0.0}"
SCHEDULER_PORT="${SCHEDULER_PORT:-8010}"
MAX_CONCURRENT="${MAX_CONCURRENT:-256}"
TIMEOUT="${TIMEOUT:-120}"
MAX_RETRIES="${MAX_RETRIES:-2}"
LOG_DIR="${LOG_DIR:-}"

REWARD_MODEL_TAG=""
REWARD_MODEL_NAME=""
LIST_TAGS=0

usage() {
  cat <<EOF
Usage:
  bash ${BASH_SOURCE[0]} --tag TAG [--model MODEL]
  bash ${BASH_SOURCE[0]} --list-tags

Examples:
  bash ${BASH_SOURCE[0]} --tag 021 --model realism
  bash ${BASH_SOURCE[0]} --tag 022 --model overall
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
    --registry)
      require_option_value "$1" "${2:-}"
      MODEL_REGISTRY_PATH="$2"
      shift 2
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

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

find_python() {
  local candidates=()
  local candidate=""

  if [[ -n "${PYTHON_BIN}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  if [[ -n "${HPSV3_ENV}" ]]; then
    candidates+=("${HPSV3_ENV}/bin/python")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi

  for candidate in "${candidates[@]}"; do
    [[ -x "${candidate}" ]] || continue
    if PYTHONPATH="${HPSV3_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${candidate}" -c \
      'import aiohttp, fastapi, torch, uvicorn; import hpsv3; import hpsv3_realism_inference; import zhiqian_model_sysprompt' \
      >/dev/null 2>&1; then
      printf '%s' "${candidate}"
      return 0
    fi
  done

  echo "Could not find a Python runtime with HPSv3 service dependencies." >&2
  echo "Set PYTHON_BIN or HPSV3_ENV and retry." >&2
  return 1
}

resolve_model() {
  if [[ -n "${ZHIQIAN_MODEL_TAG}" && -n "${ZHIQIAN_MODEL_PATH}" ]]; then
    return
  fi

  if [[ -z "${REWARD_MODEL_TAG}" ]]; then
    echo "--tag is required when ZHIQIAN_MODEL_TAG/PATH are not set." >&2
    usage >&2
    exit 2
  fi

  local lookup_tag="${REWARD_MODEL_TAG}"
  if [[ -n "${REWARD_MODEL_NAME}" ]]; then
    lookup_tag="${REWARD_MODEL_TAG}_${REWARD_MODEL_NAME}"
  fi

  local assignments=""
  if ! assignments=$("${PYTHON_BIN}" "${SCRIPT_DIR}/model_registry.py" \
      --registry "${MODEL_REGISTRY_PATH}" resolve \
      --tag "${lookup_tag}" \
      --format shell); then
    exit 1
  fi
  eval "${assignments}"
  ZHIQIAN_MODEL_TAG="${RESOLVED_MODEL_TAG}"
  ZHIQIAN_MODEL_PATH="${REGISTERED_CHECKPOINT_PATH}"
}

ensure_paths() {
  if [[ ! -d "${HPSV3_REPO}" ]]; then
    echo "HPSV3_REPO does not exist: ${HPSV3_REPO}" >&2
    exit 1
  fi
  if [[ ! -f "${HPSV3_REPO}/hpsv3_realism_inference.py" ]]; then
    echo "Missing shared inference script: ${HPSV3_REPO}/hpsv3_realism_inference.py" >&2
    exit 1
  fi
  if [[ ! -f "${HPSV3_REPO}/zhiqian_model_sysprompt.py" ]]; then
    echo "Missing system prompt registry: ${HPSV3_REPO}/zhiqian_model_sysprompt.py" >&2
    exit 1
  fi
  if [[ ! -e "${ZHIQIAN_MODEL_PATH}" ]]; then
    echo "ZHIQIAN_MODEL_PATH does not exist: ${ZHIQIAN_MODEL_PATH}" >&2
    exit 1
  fi
  if [[ ! "${ZHIQIAN_MODEL_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Invalid ZHIQIAN_MODEL_TAG: ${ZHIQIAN_MODEL_TAG}" >&2
    exit 1
  fi
  if ! ZHIQIAN_INSTRUCTION_NAME=$(PYTHONPATH="${HPSV3_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/prompt_registry.py" \
      --tag "${ZHIQIAN_MODEL_TAG}"); then
    exit 1
  fi
}

start_backend() {
  local index="$1"
  local gpu="$2"
  local port="$3"
  local log_file="${LOG_DIR}/zhiqian_${ZHIQIAN_MODEL_TAG}_gpu${gpu}_port${port}.log"

  echo "Starting Zhiqian reward backend ${index}"
  echo "  Tag        : ${ZHIQIAN_MODEL_TAG}"
  echo "  Model      : ${ZHIQIAN_MODEL_PATH}"
  echo "  Instruction: ${ZHIQIAN_INSTRUCTION_NAME}"
  echo "  GPU        : ${gpu}"
  echo "  Endpoint   : http://${HOST}:${port}"
  echo "  Log        : ${log_file}"

  (
    exec >>"${log_file}" 2>&1
    echo "[$(date '+%F %T')] Starting ${ZHIQIAN_MODEL_TAG} on GPU ${gpu}, port ${port}"
    exec env \
      PYTHONPATH="${HPSV3_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
      HPSV3_REPO="${HPSV3_REPO}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      ZHIQIAN_MODEL_TAG="${ZHIQIAN_MODEL_TAG}" \
      ZHIQIAN_MODEL_PATH="${ZHIQIAN_MODEL_PATH}" \
      HPSV3_DEVICE="${HPSV3_DEVICE}" \
      "${PYTHON_BIN}" -m uvicorn service:app \
        --app-dir "${SCRIPT_DIR}" \
        --host "${HOST}" \
        --port "${port}"
  ) &
  BACKEND_PIDS+=("$!")
}

cleanup() {
  if [[ "${#BACKEND_PIDS[@]}" -gt 0 ]]; then
    echo "Stopping Zhiqian backend processes: ${BACKEND_PIDS[*]}"
    kill "${BACKEND_PIDS[@]}" >/dev/null 2>&1 || true
  fi
}

if [[ ! -d "${HPSV3_REPO}" ]]; then
  echo "HPSV3_REPO does not exist: ${HPSV3_REPO}" >&2
  exit 1
fi
PYTHON_BIN="$(find_python)"

if [[ "${LIST_TAGS}" == "1" ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/model_registry.py" \
    --registry "${MODEL_REGISTRY_PATH}" list
fi

resolve_model
ensure_paths
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${ZHIQIAN_MODEL_TAG}}"
mkdir -p "${LOG_DIR}"

echo "Using Python: ${PYTHON_BIN}"
echo "Using shared inference: ${HPSV3_REPO}/hpsv3_realism_inference.py"
echo "Using system prompts  : ${HPSV3_REPO}/zhiqian_model_sysprompt.py"

BACKEND_PIDS=()
trap cleanup EXIT INT TERM

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

if [[ -n "${PORTS_CSV}" ]]; then
  IFS=',' read -r -a PORTS <<< "${PORTS_CSV}"
  if [[ "${#GPUS[@]}" -ne "${#PORTS[@]}" ]]; then
    echo "GPUS_CSV and PORTS_CSV must have the same number of entries." >&2
    exit 1
  fi

  BACKENDS=()
  for index in "${!GPUS[@]}"; do
    gpu="$(trim "${GPUS[$index]}")"
    port="$(trim "${PORTS[$index]}")"
    BACKENDS+=("http://127.0.0.1:${port}")
    start_backend "${index}" "${gpu}" "${port}"
  done

  if [[ "${START_SCHEDULER}" == "1" ]]; then
    BACKENDS_CSV="$(IFS=,; echo "${BACKENDS[*]}")"
    MAX_CONCURRENCY="${MAX_CONCURRENT}" \
    SCHEDULER_TIMEOUT="${TIMEOUT}" \
      "${PYTHON_BIN}" "${FLOW_GRPO_ROOT}/server/qwenvl/scheduler.py" \
        --backends "${BACKENDS_CSV}" \
        --host "${SCHEDULER_HOST}" \
        --port "${SCHEDULER_PORT}" \
        --max-retries "${MAX_RETRIES}" \
        --retry-on-5xx
  else
    echo "Started ${#BACKENDS[@]} backends without a scheduler."
    wait
  fi
else
  start_backend 0 "${GPUS_CSV}" "${PORT}"
  wait "${BACKEND_PIDS[0]}"
fi
