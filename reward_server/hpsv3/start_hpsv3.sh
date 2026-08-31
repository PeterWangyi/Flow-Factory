#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FLOW_GRPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

HPSV3_REPO="${HPSV3_REPO:-/mnt/aigc/wangyubo/code/IG/neo/RL/HPSv3}"
HPSV3_ENV="${HPSV3_ENV:-/mnt/aigc/wangyubo/anaconda3/envs/hpsv3}"
HPSV3_CONFIG_PATH="${HPSV3_CONFIG_PATH:-${HPSV3_REPO}/hpsv3/config/HPSv3_7B_service.yaml}"
HPSV3_CHECKPOINT_PATH="${HPSV3_CHECKPOINT_PATH:-/mnt/aigc/shared_env/huggingface/hub/models--MizzenAI--HPSv3/snapshots/4f81e3e09edd82fe3c5f636444c721b592a735ca/HPSv3.safetensors}"
HPSV3_DEVICE="${HPSV3_DEVICE:-cuda}"

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
LOG_DIR="${LOG_DIR:-${FLOW_GRPO_ROOT}/server/hpsv3/logs}"

PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-}"

trim() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "$x"
}

find_python() {
  local candidates=()
  local candidate=""

  if [[ -n "${PYTHON_BIN}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  if [[ -n "${VENV_DIR}" ]]; then
    candidates+=("${VENV_DIR}/bin/python")
  fi
  if [[ -n "${HPSV3_ENV}" ]]; then
    candidates+=("${HPSV3_ENV}/bin/python")
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates+=("${VIRTUAL_ENV}/bin/python")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi

  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    [[ -x "${candidate}" ]] || continue
    if (
      cd "${HPSV3_REPO}" && \
      "${candidate}" -c 'import aiohttp, fastapi, uvicorn; import hpsv3' >/dev/null 2>&1
    ); then
      printf '%s' "${candidate}"
      return 0
    fi
  done

  echo "Could not find a usable Python runtime with aiohttp, fastapi, uvicorn, and hpsv3 importable." >&2
  echo "Tried candidates:" >&2
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    echo "  ${candidate}" >&2
  done
  echo "Set PYTHON_BIN=/path/to/python or HPSV3_ENV=/path/to/conda/env and retry." >&2
  return 1
}

ensure_paths() {
  if [[ ! -d "${HPSV3_REPO}" ]]; then
    echo "HPSV3_REPO does not exist: ${HPSV3_REPO}" >&2
    exit 1
  fi
  if [[ ! -f "${HPSV3_CONFIG_PATH}" ]]; then
    echo "HPSV3_CONFIG_PATH does not exist: ${HPSV3_CONFIG_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${HPSV3_CHECKPOINT_PATH}" ]]; then
    echo "HPSV3_CHECKPOINT_PATH does not exist: ${HPSV3_CHECKPOINT_PATH}" >&2
    exit 1
  fi
}

start_backend() {
  local idx="$1"
  local gpu="$2"
  local port="$3"
  local log_file="${LOG_DIR}/hpsv3_gpu${gpu}_port${port}.log"

  echo "Starting HPSv3 service ${idx}"
  echo "  Repo  : ${HPSV3_REPO}"
  echo "  Host  : ${HOST}"
  echo "  Port  : ${port}"
  echo "  GPU   : ${gpu}"
  echo "  Log   : ${log_file}"

  (
    cd "${HPSV3_REPO}"
    exec >>"${log_file}" 2>&1
    echo "[$(date '+%F %T')] Starting HPSv3 backend on GPU ${gpu}, port ${port}"
    exec env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HPSV3_CONFIG_PATH="${HPSV3_CONFIG_PATH}" \
      HPSV3_CHECKPOINT_PATH="${HPSV3_CHECKPOINT_PATH}" \
      HPSV3_DEVICE="${HPSV3_DEVICE}" \
      "${PYTHON_BIN}" -m uvicorn hpsv3.server:app --host "${HOST}" --port "${port}"
  ) &
  BACKEND_PIDS+=("$!")
}

cleanup() {
  if [[ "${#BACKEND_PIDS[@]}" -gt 0 ]]; then
    echo "Stopping HPSv3 backend processes: ${BACKEND_PIDS[*]}"
    kill "${BACKEND_PIDS[@]}" >/dev/null 2>&1 || true
  fi
}

ensure_paths
PYTHON_BIN="$(find_python)"
mkdir -p "${LOG_DIR}"

echo "Using Python: ${PYTHON_BIN}"

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
  for idx in "${!GPUS[@]}"; do
    gpu="$(trim "${GPUS[$idx]}")"
    port="$(trim "${PORTS[$idx]}")"
    BACKENDS+=("http://127.0.0.1:${port}")
    start_backend "${idx}" "${gpu}" "${port}"
  done

  if [[ "${START_SCHEDULER}" == "1" ]]; then
    BACKENDS_CSV="$(IFS=,; echo "${BACKENDS[*]}")"
    echo "Starting scheduler"
    echo "  Host     : ${SCHEDULER_HOST}"
    echo "  Port     : ${SCHEDULER_PORT}"
    echo "  Backends : ${BACKENDS_CSV}"

    MAX_CONCURRENCY="${MAX_CONCURRENT}" \
    SCHEDULER_TIMEOUT="${TIMEOUT}" \
    "${PYTHON_BIN}" "${FLOW_GRPO_ROOT}/server/qwenvl/scheduler.py" \
      --backends "${BACKENDS_CSV}" \
      --host "${SCHEDULER_HOST}" \
      --port "${SCHEDULER_PORT}" \
      --max-retries "${MAX_RETRIES}" \
      --retry-on-5xx
  else
    echo "Started ${#BACKENDS[@]} backend services."
    echo "Scheduler not started. Use:"
    echo "${PYTHON_BIN} ${FLOW_GRPO_ROOT}/server/qwenvl/scheduler.py --backends \"$(IFS=,; echo "${BACKENDS[*]}")\" --host ${SCHEDULER_HOST} --port ${SCHEDULER_PORT} --max-retries ${MAX_RETRIES} --retry-on-5xx"
    wait
  fi
else
  echo "Starting single HPSv3 service"
  echo "  Repo  : ${HPSV3_REPO}"
  echo "  Host  : ${HOST}"
  echo "  Port  : ${PORT}"
  echo "  GPUs  : ${GPUS_CSV}"

  cd "${HPSV3_REPO}"
  CUDA_VISIBLE_DEVICES="${GPUS_CSV}" \
  HPSV3_CONFIG_PATH="${HPSV3_CONFIG_PATH}" \
  HPSV3_CHECKPOINT_PATH="${HPSV3_CHECKPOINT_PATH}" \
  HPSV3_DEVICE="${HPSV3_DEVICE}" \
  exec "${PYTHON_BIN}" -m uvicorn hpsv3.server:app --host "${HOST}" --port "${PORT}"
fi



# cd /mnt/aigc/wangyubo/code/IG/neo/flow_grpo_neo && conda activate /mnt/aigc/wangyubo/anaconda3/envs/hpsv3

# HPSV3_ENV=/mnt/aigc/wangyubo/anaconda3/envs/hpsv3 \
# GPUS_CSV=0,1,2,3,4,5,6,7 \
# PORTS_CSV=8000,8001,8002,8003,8004,8005,8006,8007 \
# START_SCHEDULER=1 \
# SCHEDULER_PORT=9010 \
# bash server/hpsv3/start_hpsv3.sh

# HPSV3_ENV=/mnt/aigc/wangyubo/anaconda3/envs/hpsv3 \
# GPUS_CSV=0,1,2,3 \
# PORTS_CSV=8000,8001,8002,8003 \
# START_SCHEDULER=1 \
# SCHEDULER_PORT=9010 \
# bash server/hpsv3/start_hpsv3.sh


# cd /mnt/aigc/wangyubo/code/IG/neo/flow_grpo_neo

# HPSV3_ENV=/mnt/aigc/wangyubo/anaconda3/envs/hpsv3 \
# GPUS_CSV=0,1 \
# PORTS_CSV=9000,9001 \
# START_SCHEDULER=1 \
# SCHEDULER_PORT=9010 \
# bash server/hpsv3/start_hpsv3.sh
