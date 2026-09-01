# 美学模型实验启动指南

本文以 **Z-Image** 为例，说明美学模型实验的启动流程。以下命令默认在
`/mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory` 目录下执行。

## 1. 启动 Reward Server

### 1.1 启动 HPSv3 Server

启动脚本为 `reward_server/hpsv3/submit_hps_multinode_1gpu.sh`。

修改脚本中的目标集群和 GPU 卡数后，执行：

```bash
bash reward_server/hpsv3/submit_hps_multinode_1gpu.sh
```

### 1.2 启动 Zhiqian Reward Model

启动脚本为 `reward_server/zhiqian_aeth_model/submit_multinode_1gpu.sh`。

接入新的 Reward Model（RM）时，需要完成以下配置：

1. 在 `reward_server/zhiqian_aeth_model/models.json` 中注册模型。
2. 在 `/mnt/aigc/wangyubo/code/IG/neo/RL/HPSv3/zhiqian_model_sysprompt.py` 中添加对应的 system prompt。

例如，启动 `v021 realism` RM：

```bash
bash reward_server/zhiqian_aeth_model/submit_multinode_1gpu.sh \
  --tag 021 \
  --model realism
```

## 2. 配置训练任务

编辑配置文件：

```text
examples/grpo/lora/z_image/hpsv3_2x8.yaml
```

重点修改以下字段：

- `dataset_dir`：数据集目录
- `cache_dir`：缓存目录。为预先存放text embedding的位置，注意对比实验启动时，不能同时写入，注意时间间隔。
- `run_name`：实验名称
- `save_dir`：模型及训练产物的保存目录
- `server_url`：步骤 1 中启动的 Reward Server 地址。任务启动后，在算力池中找到该任务，并填写 master 节点所在机器的 `addr`

## 3. 提交训练任务

完成配置后，使用以下脚本提交训练任务：

```bash
bash peter_training/test_scripts/submit.sh
```

## 其他工具

Rollout 工具位于：

```text
/mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory/tools/rollout
```
