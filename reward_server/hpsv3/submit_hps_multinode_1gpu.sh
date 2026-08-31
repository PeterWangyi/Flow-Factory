#!/usr/bin/env bash
set -euo pipefail

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
AEC2_NAME="vigen"
# AEC2_NAME="m-train-neo2-infographic"
# AEC2_NAME="neo1-edit"
# AEC2_NAME="neo1-trajectory"
# AEC2_NAME="neo1-agentic"
# AEC2_NAME="si"
# AEC2_NAME="umm"

### job name
# JOB_NAME="neo_infog_hps_rl"
# JOB_NAME="neo_infog_eval"
JOB_NAME="U15-InfoV1_reward_server-hpsv3"
# JOB_NAME="U1-InfoV3_DPO"

### storage mount
STORAGE_MOUNT="1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs,c83d08bc-2965-11ef-b8c5-929f74fd8884:/mnt/aigc/,01998fb1-b876-7b33-82c9-4427517bf536:/mnt/umm"

### worker spec
WORKER_NODES=1
# WORKER_SPEC="N6lS.Iu.I80.1"  # default
WORKER_SPEC="N6lS.Iq.I10.1"  # vigen si

HPSV3_ENV="${HPSV3_ENV:-/mnt/aigc/wangyubo/anaconda3/envs/hpsv3}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
SCHEDULER_PORT="${SCHEDULER_PORT:-9010}"

# The cluster runs this same command on every worker and injects
# RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT. Only rank 0 exposes the scheduler.
COMMAND="cd /mnt/aigc/wangyubo/code/IG/neo/flow_grpo_neo && \
HPSV3_ENV=${HPSV3_ENV} \
BACKEND_PORT=${BACKEND_PORT} \
SCHEDULER_PORT=${SCHEDULER_PORT} \
bash server/hpsv3/start_hpsv3_multinode_1gpu.sh
"

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
