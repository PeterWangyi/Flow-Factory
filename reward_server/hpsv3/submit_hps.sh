#!/usr/bin/env bash
set -euo pipefail

### common
WORKSPACE_NAME="aigc"
PRIORITY="HIGHEST"
TRAINING_FRAMEWORK="pytorch"
# COMMAND="sleep inf"
# COMMAND="sleep 1d"

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
# JOB_NAME="neo_infog_hps_rl"
# JOB_NAME="neo_infog_eval"
JOB_NAME="U15-InfoV0_reward_server-hps"
# JOB_NAME="U1-InfoV3_DPO"

# JOB_NAME="neo_u11_ocr_reward_server"
# JOB_NAME="neo_u1_infog_eval"
# JOB_NAME="neo_u1_DPO"
# JOB_NAME="neo_debug"
# JOB_NAME="neo_info_hps_grpo"
# JOB_NAME="neo_u1_info1.1_ocr_stage"
# JOB_NAME="neo_u1_info1.1_stage2"
# JOB_NAME="neo_u11_info_dpo-debug"

# JOB_NAME="U11_Info_DPO_donot_kill_run_24h"

### storage mount
STORAGE_MOUNT="1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs,c83d08bc-2965-11ef-b8c5-929f74fd8884:/mnt/aigc/,01998fb1-b876-7b33-82c9-4427517bf536:/mnt/umm"

### worker spec
WORKER_NODES=1
WORKER_SPEC="N6lS.Iu.I80.8"  # default
# WORKER_SPEC="N6lS.Iq.I10.8"  # vigen si

COMMAND="cd /mnt/aigc/wangyubo/code/IG/neo/flow_grpo && \
HPSV3_ENV=/mnt/aigc/wangyubo/anaconda3/envs/hpsv3 \
GPUS_CSV=0,1,2,3,4,5,6,7 \
PORTS_CSV=8000,8001,8002,8003,8004,8005,8006,8007 \
START_SCHEDULER=1 \
SCHEDULER_PORT=9010 \
bash server/hpsv3/start_hpsv3.sh
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
