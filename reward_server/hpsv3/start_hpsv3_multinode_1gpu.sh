#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FLOW_GRPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

NODE_RANK="${RANK:-${NODE_RANK:-}}"
NUM_NODES="${WORLD_SIZE:-${NNODES:-}}"
MASTER_ADDR_VALUE="${MASTER_ADDR:-}"
RENDEZVOUS_PORT="${REWARD_RENDEZVOUS_PORT:-${MASTER_PORT:-}}"

GPU_ID="${GPU_ID:-0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
SCHEDULER_HOST="${SCHEDULER_HOST:-0.0.0.0}"
SCHEDULER_PORT="${SCHEDULER_PORT:-9010}"
MAX_CONCURRENT="${MAX_CONCURRENT:-256}"
TIMEOUT="${TIMEOUT:-120}"
MAX_RETRIES="${MAX_RETRIES:-2}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
REWARD_ADVERTISE_HOST="${REWARD_ADVERTISE_HOST:-}"

HPSV3_ENV="${HPSV3_ENV:-/mnt/aigc/wangyubo/anaconda3/envs/hpsv3}"
COORDINATOR_PYTHON="${COORDINATOR_PYTHON:-${HPSV3_ENV}/bin/python}"

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "Missing ${name}. This launcher expects the cluster to inject RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT." >&2
    exit 1
  fi
}

require_value "RANK (or NODE_RANK)" "${NODE_RANK}"
require_value "WORLD_SIZE (or NNODES)" "${NUM_NODES}"
require_value "MASTER_ADDR" "${MASTER_ADDR_VALUE}"
require_value "MASTER_PORT (or REWARD_RENDEZVOUS_PORT)" "${RENDEZVOUS_PORT}"

if [[ ! "${NODE_RANK}" =~ ^[0-9]+$ ]] || [[ ! "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RANK and WORLD_SIZE must be integers; got RANK=${NODE_RANK}, WORLD_SIZE=${NUM_NODES}." >&2
  exit 1
fi
if (( NODE_RANK >= NUM_NODES )); then
  echo "RANK must be smaller than WORLD_SIZE; got RANK=${NODE_RANK}, WORLD_SIZE=${NUM_NODES}." >&2
  exit 1
fi
if [[ ! -x "${COORDINATOR_PYTHON}" ]]; then
  echo "COORDINATOR_PYTHON is not executable: ${COORDINATOR_PYTHON}" >&2
  exit 1
fi
if [[ "${BACKEND_PORT}" == "${SCHEDULER_PORT}" ]]; then
  echo "BACKEND_PORT and SCHEDULER_PORT must differ on rank 0." >&2
  exit 1
fi
if [[ "${BACKEND_PORT}" == "${RENDEZVOUS_PORT}" ]]; then
  echo "BACKEND_PORT and MASTER_PORT/REWARD_RENDEZVOUS_PORT must differ on rank 0." >&2
  exit 1
fi

backend_pid=""
cleanup() {
  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" >/dev/null 2>&1; then
    echo "Stopping HPSv3 backend PID ${backend_pid} on rank ${NODE_RANK}"
    kill "${backend_pid}" >/dev/null 2>&1 || true
    wait "${backend_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting one-GPU HPSv3 backend on rank ${NODE_RANK}/${NUM_NODES}"
echo "  GPU             : ${GPU_ID}"
echo "  Backend         : 0.0.0.0:${BACKEND_PORT}"
echo "  Rendezvous      : ${MASTER_ADDR_VALUE}:${RENDEZVOUS_PORT}"
if [[ "${NODE_RANK}" == "0" ]]; then
  echo "  Scheduler       : ${SCHEDULER_HOST}:${SCHEDULER_PORT}"
fi

HPSV3_ENV="${HPSV3_ENV}" \
GPUS_CSV="${GPU_ID}" \
PORT="${BACKEND_PORT}" \
PORTS_CSV="" \
START_SCHEDULER=0 \
  bash "${SCRIPT_DIR}/start_hpsv3.sh" &
backend_pid=$!

set +e
BACKENDS_CSV=$(
  "${COORDINATOR_PYTHON}" "${FLOW_GRPO_ROOT}/server/multinode/collect_backends.py" \
    --rank "${NODE_RANK}" \
    --world-size "${NUM_NODES}" \
    --master-addr "${MASTER_ADDR_VALUE}" \
    --master-port "${RENDEZVOUS_PORT}" \
    --backend-port "${BACKEND_PORT}" \
    --health-url "http://127.0.0.1:${BACKEND_PORT}/healthz" \
    --advertise-host "${REWARD_ADVERTISE_HOST}" \
    --health-timeout "${STARTUP_TIMEOUT}" \
    --rendezvous-timeout "${STARTUP_TIMEOUT}"
)
rendezvous_status=$?
set -e

if (( rendezvous_status != 0 )); then
  exit "${rendezvous_status}"
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  echo "All ${NUM_NODES} HPSv3 backends are ready."
  echo "Backends: ${BACKENDS_CSV}"
  echo "Unified endpoint: http://${MASTER_ADDR_VALUE}:${SCHEDULER_PORT}"

  MAX_CONCURRENCY="${MAX_CONCURRENT}" \
  SCHEDULER_TIMEOUT="${TIMEOUT}" \
    "${COORDINATOR_PYTHON}" "${FLOW_GRPO_ROOT}/server/qwenvl/scheduler.py" \
      --backends "${BACKENDS_CSV}" \
      --host "${SCHEDULER_HOST}" \
      --port "${SCHEDULER_PORT}" \
      --max-retries "${MAX_RETRIES}" \
      --retry-on-5xx
else
  echo "Rank ${NODE_RANK} backend registered; waiting for it to exit."
  wait "${backend_pid}"
fi
