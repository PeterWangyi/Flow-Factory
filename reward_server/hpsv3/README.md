# HPSv3 服务

HPSv3 reward 服务依赖外部 HPSv3 仓库；本目录提供启动封装、连通性测试和离线结果查看工具。

## 准备

至少需要设置正确的仓库、配置和权重路径：

```bash
export HPSV3_REPO=/path/to/HPSv3
export HPSV3_CONFIG_PATH=$HPSV3_REPO/hpsv3/config/HPSv3_7B_service.yaml
export HPSV3_CHECKPOINT_PATH=/path/to/HPSv3.safetensors
export HPSV3_ENV=/path/to/hpsv3-env
```

## 启动

单卡默认监听 `8010`：

```bash
GPUS_CSV=0 PORT=8010 bash server/hpsv3/start_hpsv3.sh
```

多卡统一出口：

```bash
GPUS_CSV=0,1,2,3 \
PORTS_CSV=8000,8001,8002,8003 \
START_SCHEDULER=1 SCHEDULER_PORT=8010 \
  bash server/hpsv3/start_hpsv3.sh
```

backend 提供 `GET /healthz`、`POST /score` 和 `POST /scores`；scheduler 提供 `GET /health`、`GET /backends`，并代理打分接口。

连通性测试：

```bash
python server/hpsv3/test_hpsv3_service.py \
  --url http://127.0.0.1:8010 \
  --image /path/to/image.png \
  --prompt "a cat sitting on a chair"
```

## 训练配置

```yaml
hpsv3_base_url: "http://<server-ip>:8010"
hpsv3_timeout: 120
hpsv3_batch_size: 16

reward_fn:
  hpsv3: 1.0
```

若使用本地 OCR 惩罚版本：

```yaml
ocr_penalty_beta: 10
ocr_penalty_weight: 0.1

reward_fn:
  hpsv3_OCRpenalty: 1.0
```

其奖励为 `hpsv3_raw + ocr_penalty_beta * ocr_penalty`；OCR 在训练进程内执行，不由 HPSv3 服务处理。

