#!/usr/bin/env bash
set -euo pipefail

### common
WORKSPACE_NAME="aigc"
PRIORITY="HIGHEST"
TRAINING_FRAMEWORK="pytorch"
# COMMAND="sleep inf"
# COMMAND="sleep 5d"

# container image
CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/ccr_2/ulimit-change:20240725-18h11m01s"
# CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/ccr_2/wan21-v4:20250325-16h55m11s"      # qwen ocr 
# CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/ccr_2/pytorch:2.5.1-cuda12.4-cudnn9-devel"
# CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/ccr_2/fvg_base_25:20250427_43b214e"  # with vllm
# CONTAINER_IMAGE_URL="registry.ms-sc-01.maoshanwangtech.com/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.28.0:master-20230626-172512-32302"  # u15

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
# AEC2_NAME="m-train-neo3"

### job name
JOB_NAME="U15-Info-Rollout"
# JOB_NAME="CKPT"

### storage mount
STORAGE_MOUNT="1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs,c83d08bc-2965-11ef-b8c5-929f74fd8884:/mnt/aigc,01998fb1-b876-7b33-82c9-4427517bf536:/mnt/umm,019d4c4c-c68d-7cf9-a5af-d5856d0c457e:/mnt/afs-openrouter"

### worker spec
WORKER_NODES=1
WORKER_SPEC="N6lS.Iu.I80.2"  # default
# WORKER_SPEC="N6lS.Iq.I10.1"  # vigen si


CONFIGS=(
  '/mnt/aigc/tuyouyuan/code/Flow-Factory/tools/rollout/configs/peter/zoe_inference/z_image.yaml'
#   '/mnt/aigc/tuyouyuan/code/Flow-Factory/tools/rollout/configs/peter/zoe_inference/flux2_klein.yaml'
#   '/mnt/aigc/tuyouyuan/code/Flow-Factory/tools/rollout/configs/peter/zoe_inference/qwen_image.yaml'
)



for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
    COMMAND="source /mnt/aigc/wangyubo/anaconda3/etc/profile.d/conda.sh && conda activate /mnt/aigc/wangyubo/anaconda3/envs/flowfactory && cd /mnt/aigc/tuyouyuan/code/Flow-Factory && python tools/rollout/rollout.py --config ${CONFIG}"

    sco acp jobs create \
        --workspace-name="${WORKSPACE_NAME}" \
        --aec2-name="${AEC2_NAME}" \
        --job-name="${JOB_NAME}-${CONFIG_NAME}" \
        --priority="${PRIORITY}" \
        --container-image-url="${CONTAINER_IMAGE_URL}" \
        --storage-mount="${STORAGE_MOUNT}" \
        --training-framework="${TRAINING_FRAMEWORK}" \
        --worker-nodes="${WORKER_NODES}" \
        --worker-spec="${WORKER_SPEC}" \
        --command="${COMMAND}"
done