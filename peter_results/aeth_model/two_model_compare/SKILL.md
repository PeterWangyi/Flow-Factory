---
name: flow-factory-two-model-compare
description: Build a static HTML report that compares the latest complete W&B eval image record from two Flow-Factory save directories. Resolves centralized repository-level wandb runs from each save directory's run_name, auto-detects eval image keys, pairs images by exact prompt, and references original /mnt media without copying it.
---

# Flow-Factory Two Model Compare

Use `scripts/build_two_model_compare.py` to compare the latest complete eval images from two
Flow-Factory runs. The inputs may be save directories, W&B run directories, or `run-*.wandb`
files.

## Workflow

1. Pass the two save directories with `--run-a` and `--run-b`.
2. The script resolves a save directory to the repository-level `wandb/run-*` directory by
   matching its basename against `run_name` in W&B `files/config.yaml`.
3. Unless `--image-key` is provided, it selects the newest complete history record whose image
   key contains `eval` and has equal, non-empty `filenames` and `captions` arrays.
4. It accepts both Flow-Factory captions (`score | prompt`) and the older
   `metric: score | avg: value | prompt` format.
5. It pairs the two models by exact cleaned prompt text and fails on missing or duplicate prompts.
6. It writes one static HTML file. Images remain in W&B media and are referenced using absolute
   root-relative `/mnt/...` paths.

## Command

Use the Python environment that ran Flow-Factory so the `wandb` package is available:

```bash
/path/to/flowfactory/bin/python scripts/build_two_model_compare.py \
  --run-a /absolute/path/to/first-save-directory \
  --run-b /absolute/path/to/second-save-directory \
  --label-a HPSv3 \
  --label-b "v020 realism" \
  --title "Z-Image latest eval comparison" \
  --output /absolute/path/to/latest_eval_compare.html
```

Use `--image-key eval/aesthetics/samples` only when automatic discovery is undesirable.

## Validation

The JSON printed by the script must show:

- the expected W&B file for each save run;
- the same non-zero image count for both models;
- the newest complete eval step and selected image key;
- `missing_images: 0` for both models.

The report deliberately describes cross-model score differences as raw deltas. Do not claim that
scores from different reward models are calibrated to the same scale.
