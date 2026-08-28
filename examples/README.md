# Examples

Training configs for all supported algorithm–model combinations.

## Directory Structure

```
examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml
```

| Level | Description | Examples |
|-------|-------------|---------|
| `algorithm` | Training algorithm | `grpo`, `dppo`, `nft`, `awm`, `dgpo`, `dpo`, `crd`, `opd`, `dmd2`, `tdm`, `tdm_r1` |
| `finetune_type` | Parameter-efficient or full | `lora`, `full` |
| `model_type` | Model family (underscore-separated) | `flux1`, `sd3_5`, `wan21`, `ltx2` |
| `variant` | Config variant | `default.yaml`, `nocfg.yaml`, `t2v.yaml` |

**Naming rules**:
- Model directory names use underscores matching the config's `model_type` field (e.g., `sd3-5` → `sd3_5`, `flux1-kontext` → `flux1_kontext`).
- `default.yaml` is the baseline config for a model. Use descriptive names for variants (`nocfg.yaml`, `rational_rewards_t2i.yaml`, `t2v.yaml`, `i2v.yaml`).

**Quick start**:
```bash
ff-train examples/grpo/lora/flux1/default.yaml
```

## Two-node HPSv3 GRPO recipes

The following starting points target two 8-GPU, 80 GB nodes and an HPSv3
service exposing `POST /score`:

- [`FLUX.2-dev`](grpo/lora/flux2/hpsv3_2x8.yaml): FSDP2, 8-step training
  rollouts, and 28-step evaluation.
- [`Qwen-Image-2512`](grpo/lora/qwen_image/hpsv3_2x8.yaml): FSDP2, 10-step
  training rollouts, and 50-step evaluation.
- [`Z-Image`](grpo/lora/z_image/hpsv3_2x8.yaml): DeepSpeed ZeRO-2, 10-step
  training rollouts, and 40-step evaluation.
- [`SD3.5 Medium`](grpo/lora/sd3_5/hpsv3_2x8_medium.yaml): DeepSpeed ZeRO-2,
  10-step training rollouts, and 40-step evaluation.

First create the compact, uniformly sampled 8192-prompt dataset. This writes a
new directory and leaves the source files untouched:

```bash
python scripts/prepare_aesthetics_prompts.py \
  --input-dir /mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory/peter_data/dataset/aesthetics/filter-hps-std-gt-0.4 \
  --output-dir /mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory/peter_data/dataset/aesthetics/filter-hps-std-gt-0.4-caption-8192 \
  --train-limit 8192 \
  --seed 42
```

Use the shared launcher on both nodes. Pass the same YAML path, but use rank 0
on the master and rank 1 on the second node:

```bash
# Master node
bash peter_training/test_scripts/launch_hpsv3_2x8.sh examples/grpo/lora/qwen_image/hpsv3_2x8.yaml 0

# Second node
bash peter_training/test_scripts/launch_hpsv3_2x8.sh examples/grpo/lora/qwen_image/hpsv3_2x8.yaml 1
```

Use any other recipe by replacing the YAML path. Environment-only invocation is
also supported through `FF_CONFIG` and `MACHINE_RANK`. Set `FF_LOG_DIR` to change
the per-node log directory; extra positional arguments after the machine rank
are forwarded to `ff-train`.

The dataset path must exist on both nodes. With local preprocessing, the cache
need not be shared. Before a long run, verify `/healthz` or `/health` and one
`/score` request from each training node, then use a one-epoch smoke run.

## DMD2 and TDM

- [`dmd2` SD3.5 OCR recipe](dmd2/lora/sd3_5/ocr.yaml) — validated in a
  distributed OCR training run.
- [`tdm` SD3.5 OCR recipe](tdm/lora/sd3_5/ocr.yaml) — official conditional-noise objective.
- [`tdm-r1` SD3.5 OCR recipe](tdm_r1/lora/sd3_5/ocr.yaml) — official G24
  fake-surrogate-generator objective initialized from the released TDM adapter.

## MiniMax H3 examples

- [`minimax-h3-t2va`](../examples/grpo/lora/minimax_h3_t2va/default.yaml)
- [`minimax-h3-t2va` real-weight debug recipe](../examples/grpo/lora/minimax_h3_t2va/debug.yaml)
- [`minimax-h3-t2va` native-quality FSDP2 recipe](../examples/grpo/lora/minimax_h3_t2va/quality_720p_fsdp2.yaml)
- [`minimax-h3-fl2va`](../examples/grpo/lora/minimax_h3_fl2va/default.yaml)
- [`minimax-h3-ref2va`](../examples/grpo/lora/minimax_h3_ref2va/default.yaml)

The T2VA `debug.yaml` recipe is real-weight validated with the 61 GB checkpoint
(61.74 GiB transformer):
1 GPU and 16 GPUs across two nodes completed CPS rollout, video/audio decode,
CLAP reward, GRPO replay/backward/optimizer step, and LoRA checkpoint save/resume.
Its 64x96 canvas is intentionally a correctness geometry. The quality-oriented T2VA
default remains an unverified quality starting point. FL2VA and Ref2VA are
**Schema/API validated only** rather than claims of training stability or reward
improvement.

The T2VA `quality_720p_fsdp2.yaml` recipe is the active native-quality path:
768x1344, 124 frames, 24 denoising steps, LoRA rank 64 / alpha 128, and two
updates from 48 prompt groups per epoch. Its real-weight FSDP2 initialization,
checkpoint, native-resolution decode, and CLAP evaluation are validated; a
completed long-run reward trend is not yet claimed.

FL2VA and Ref2VA use Meta ImageBind for audio-video alignment. Install ImageBind
and PyTorchVideo from their upstream repositories before running those examples;
ImageBind is licensed CC-BY-NC-SA 4.0 (NonCommercial).

```bash
pip install git+https://github.com/facebookresearch/ImageBind.git
pip install git+https://github.com/facebookresearch/pytorchvideo.git
```

## Contributing

We welcome community contributions! Here's what you can contribute and how:

### Verified Training Configs

If you've tested a model–algorithm combination and confirmed reward improvement, submit a PR with:
- The config YAML following the directory structure above
- A brief note in the PR description about hardware used and observed reward trend

> **Example**: [#145 — LTX-2.3 + PickScore](https://github.com/X-GenGroup/Flow-Factory/pull/145) added a GRPO + LoRA config for text-to-audio-video, with a training curve (8×H200, 18h) confirming reward improvement.

### Custom Reward Models

New reward models are welcome — add the implementation under `src/flow_factory/rewards/` and include an example config that uses it. Please ensure your reward model's dependencies are compatible with the existing environment (check `pyproject.toml`).

### New Model Adapters

See the [New Model Guide](../guidance/new_model.md) for how to add a new diffusion/flow-matching model. Include at least one example config with your PR.

### Guidelines

- Configs should be self-contained and runnable with `ff-train`
- Include comments for non-obvious parameter choices
- If your config requires a specific dataset, document how to obtain it
- Test on at least one hardware configuration before submitting
