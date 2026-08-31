# Zhiqian Aesthetic Reward Model Service

本目录专门用于启动 Zhiqian 训练并导出的 HPSv3 reward model，不使用官方 `hpsv3.server`。

模型共用 `${HPSV3_REPO}/hpsv3_realism_inference.py` 中的加载和推理逻辑；服务根据模型 tag，从 `${HPSV3_REPO}/zhiqian_model_sysprompt.py` 自动选择 system prompt。默认 HPSv3 仓库是 `/mnt/aigc/wangyubo/code/IG/neo/RL/HPSv3`。

## 模型注册

只需维护 `models.json`，每条记录严格包含 `tag` 和 `model_path`：

```json
[
  {
    "tag": "021_realism",
    "model_path": "/mnt/aigc/linzhiqian/model_exports/hpsv3_v021_realism/merged/HPSv3_v0.2.1_realism_merged.safetensors"
  },
  {
    "tag": "022_overall",
    "model_path": "/mnt/aigc/linzhiqian/model_exports/hpsv3_v022_overall/merged/HPSv3_v0.2.2_overall_merged.safetensors"
  }
]
```

tag 的最后一段决定 instruction：`021_realism` 对应 `REALISM_INSTRUCTION`，`022_overall` 对应 `OVERALL_INSTRUCTION`。以后新增 `023_aesthetic` 时，只需在 `zhiqian_model_sysprompt.py` 增加：

```python
AESTHETIC_INSTRUCTION = "..."
```

查看模型：

```bash
bash server/zhiqian_aeth_model/start_model.sh --list-tags
```

## 本地单卡

```bash
GPUS_CSV=0 PORT=8010 \
  bash server/zhiqian_aeth_model/start_model.sh \
  --tag 021 \
  --model realism
```

启动 overall：

```bash
GPUS_CSV=0 PORT=8010 \
  bash server/zhiqian_aeth_model/start_model.sh \
  --tag 022 \
  --model overall
```

## 本地多卡统一出口

```bash
GPUS_CSV=0,1,2,3 \
PORTS_CSV=8000,8001,8002,8003 \
START_SCHEDULER=1 \
SCHEDULER_PORT=8010 \
  bash server/zhiqian_aeth_model/start_model.sh \
  --tag 021 \
  --model realism
```

## 提交跨节点 1×N 卡任务

先 dry-run：

```bash
bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh \
  --tag 021 \
  --model realism \
  --nodes 4 \
  --dry-run
```

实际提交：

```bash
bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh \
  --tag 021 \
  --model realism \
  --nodes 4
```

提交 overall：

```bash
bash server/zhiqian_aeth_model/submit_multinode_1gpu.sh \
  --tag 022 \
  --model overall \
  --nodes 4
```

集群在每个节点启动一个 GPU backend，rank 0 启动统一 scheduler。默认 backend 端口为 `8000`，scheduler 端口为 `9010`。

## HTTP API

Backend 提供：

- `GET /healthz`
- `POST /score`
- `POST /scores`

请求格式与官方 HPSv3 服务一致。`prompt` 只传原始生图 prompt；对应模型的 instruction 由服务自动添加。

日志默认写入 `server/zhiqian_aeth_model/logs/<tag>/`。
