# JSONL Rollout

Use one YAML-first entry point for FLUX.2 Klein, Qwen-Image, and Z-Image batch inference:

```bash
python tools/rollout/rollout.py --config <rollout.yaml>
```

The launcher validates the job, selects the corresponding worker under `tools/rollout/workers`,
exposes one configured GPU, and forwards all model, data, and sampling arguments. Each prompt uses
the same configured seed; the default is `42`.

Run a validation-only pass before loading a model:

```bash
python tools/rollout/rollout.py --config <rollout.yaml> --dry-run
```

Override individual values without editing the YAML:

```bash
python tools/rollout/rollout.py \
  --config <rollout.yaml> \
  --set launcher.gpus='[1]' \
  --set data.limit=10
```

Configuration precedence is `--set` > YAML > model-specific defaults. The current workers run on
one GPU, so `launcher.gpus` must contain exactly one device index.

Output images use their zero-based source-record index, such as `000000.png`. The output directory
also contains `metadata.jsonl` with the normalized prompt, seed, model, checkpoint, resolution, and
sampling settings. Existing outputs are rejected unless `overwrite: true` is configured.
