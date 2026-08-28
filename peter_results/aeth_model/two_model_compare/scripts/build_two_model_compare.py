#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create a static comparison of the latest eval images from two Flow-Factory runs."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


LEGACY_CAPTION_RE = re.compile(
    r"^([^:|]+):\s*(-?\d+(?:\.\d+)?)\s*\|\s*"
    r"avg:\s*(-?\d+(?:\.\d+)?)\s*\|\s*(.*)$",
    re.S,
)
FLOW_FACTORY_CAPTION_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*\|\s*(.*)$", re.S)
RUN_NAME_RE = re.compile(r"^\s*run_name:\s*(['\"]?)(.*?)\1\s*$", re.M)


def esc(value: object) -> str:
    """Escape a value for HTML text or attribute use."""
    return html.escape(str(value), quote=True)


def read_config_run_name(config_path: Path) -> Optional[str]:
    """Read the W&B run_name from a generated config without requiring PyYAML."""
    match = RUN_NAME_RE.search(config_path.read_text(encoding="utf-8"))
    return match.group(2) if match else None


def repository_roots(path: Path) -> List[Path]:
    """Return ancestor directories that contain a repository-level W&B directory."""
    roots = []
    for candidate in (path, *path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "wandb").is_dir() and candidate not in roots:
            roots.append(candidate)
    return roots


def find_wandb_file(value: str) -> Path:
    """Resolve a save directory, W&B directory, or W&B data file."""
    path = Path(value).expanduser().resolve()
    if path.is_file() and path.suffix == ".wandb":
        return path
    if not path.exists():
        raise FileNotFoundError(path)

    local_candidates = list(path.glob("run-*.wandb"))
    local_candidates += list(path.glob("run-*/run-*.wandb"))
    local_candidates += list(path.glob("wandb/run-*/run-*.wandb"))
    if local_candidates:
        return max(local_candidates, key=lambda item: item.stat().st_mtime_ns)

    run_name = path.name
    matched = []
    for root in repository_roots(path):
        for config_path in (root / "wandb").glob("run-*/files/config.yaml"):
            if read_config_run_name(config_path) != run_name:
                continue
            matched.extend(config_path.parent.parent.glob("run-*.wandb"))
    if not matched:
        raise FileNotFoundError(
            f"No W&B file found for {path}. Expected a local run-*.wandb file or a repository "
            f"wandb/run-* config with run_name={run_name!r}."
        )
    return max(matched, key=lambda item: item.stat().st_mtime_ns)


def history_values(record: wandb_internal_pb2.Record) -> Dict[str, object]:
    """Decode a W&B history record into slash-separated keys."""
    values: Dict[str, object] = {}
    for item in record.history.item:
        key = "/".join(item.nested_key) if item.nested_key else item.key
        try:
            values[key] = json.loads(item.value_json)
        except json.JSONDecodeError:
            values[key] = item.value_json
    return values


def latest_complete_eval(wandb_file: Path, requested_key: Optional[str]) -> Dict[str, object]:
    """Find the newest complete eval image record, tolerating a partial live-run tail."""
    with tempfile.TemporaryDirectory(prefix="flow-factory-two-model-compare-") as tmpdir:
        snapshot = Path(tmpdir) / wandb_file.name
        shutil.copyfile(wandb_file, snapshot)
        store = DataStore()
        store.open_for_scan(str(snapshot))
        candidates = []
        while True:
            try:
                payload = store.scan_data()
            except (AssertionError, EOFError, ValueError):
                break
            if payload is None:
                break
            record = wandb_internal_pb2.Record()
            try:
                record.ParseFromString(payload)
            except Exception:
                break
            if record.WhichOneof("record_type") != "history":
                continue
            values = history_values(record)
            for filenames_key, filenames in values.items():
                if not filenames_key.endswith("/filenames"):
                    continue
                image_key = filenames_key[: -len("/filenames")]
                if requested_key and image_key != requested_key:
                    continue
                if not requested_key and "eval" not in image_key.lower():
                    continue
                captions = values.get(f"{image_key}/captions")
                if (
                    isinstance(filenames, list)
                    and isinstance(captions, list)
                    and filenames
                    and len(filenames) == len(captions)
                ):
                    candidates.append(
                        {
                            "step": int(values.get("_step", -1)),
                            "image_key": image_key,
                            "filenames": filenames,
                            "captions": captions,
                        }
                    )
    if not candidates:
        key_description = requested_key or "an automatically discovered eval image key"
        raise RuntimeError(f"No complete {key_description} history found in {wandb_file}")
    return max(candidates, key=lambda item: int(item["step"]))


def clean_prompt(prompt: str) -> str:
    """Remove common chat-template wrappers while preserving prompt contents."""
    value = prompt.strip()
    value = re.sub(r"^<\|im_start\|>user\s*", "", value)
    value = re.sub(r"<\|im_end\|>\s*<\|im_start\|>assistant\s*$", "", value)
    return value.strip()


def parse_caption(caption: str, default_metric: str) -> Tuple[str, float, str]:
    """Parse legacy and current Flow-Factory eval captions."""
    legacy_match = LEGACY_CAPTION_RE.match(caption)
    if legacy_match:
        return legacy_match.group(1).strip(), float(legacy_match.group(2)), clean_prompt(
            legacy_match.group(4)
        )
    flow_factory_match = FLOW_FACTORY_CAPTION_RE.match(caption)
    if flow_factory_match:
        return default_metric, float(flow_factory_match.group(1)), clean_prompt(
            flow_factory_match.group(2)
        )
    raise ValueError(f"Unexpected eval caption format: {caption[:180]!r}")


def metric_from_image_key(image_key: str) -> str:
    """Derive a readable metric name from an eval image key."""
    parts = [part for part in image_key.split("/") if part]
    if len(parts) >= 2 and parts[-1] in {"samples", "images"}:
        return parts[-2]
    return parts[-1] if parts else "score"


def load_model(
    run: str, label: str, color: str, requested_key: Optional[str]
) -> Dict[str, object]:
    """Load the latest complete eval record and validate its media files."""
    wandb_file = find_wandb_file(run)
    record = latest_complete_eval(wandb_file, requested_key)
    media_root = wandb_file.parent / "files"
    metric_name = metric_from_image_key(str(record["image_key"]))
    items = []
    for index, (filename, caption) in enumerate(zip(record["filenames"], record["captions"])):
        metric, score, prompt = parse_caption(str(caption), metric_name)
        source = (media_root / str(filename)).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        items.append(
            {
                "index": index,
                "prompt": prompt,
                "metric": metric,
                "score": score,
                "url": source.as_posix(),
                "label": label,
                "color": color,
            }
        )
    return {
        "label": label,
        "color": color,
        "step": record["step"],
        "image_key": record["image_key"],
        "items": items,
        "wandb_file": wandb_file,
    }


def image_card(item: Dict[str, object], rank: Optional[int] = None) -> str:
    """Render one image and its score metadata."""
    badge = f'<span class="rank-badge">#{rank}</span>' if rank else ""
    return f"""
    <article class="image-card">
      <div class="image-wrap">{badge}<a href="{esc(item['url'])}" target="_blank">
        <img src="{esc(item['url'])}" alt="Eval image {int(item['index']) + 1}" loading="lazy">
      </a></div>
      <div class="card-meta"><span class="dot" style="--dot:{esc(item['color'])}"></span>
        <span class="model-name">{esc(item['label'])}</span>
        <span class="score">{float(item['score']):.2f}</span>
      </div>
    </article>"""


def pair_items(
    model_a: Dict[str, object], model_b: Dict[str, object]
) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    """Pair unique records by exact prompt and reject mismatched prompt sets."""
    items_a = model_a["items"]
    items_b = model_b["items"]
    assert isinstance(items_a, list) and isinstance(items_b, list)
    prompts_a = Counter(str(item["prompt"]) for item in items_a)
    prompts_b = Counter(str(item["prompt"]) for item in items_b)
    duplicates = [prompt for prompt, count in (prompts_a + prompts_b).items() if count > 2]
    if duplicates:
        raise RuntimeError(f"Duplicate eval prompt cannot be paired safely: {duplicates[0][:160]}")
    if prompts_a != prompts_b:
        missing_b = list((prompts_a - prompts_b).elements())
        missing_a = list((prompts_b - prompts_a).elements())
        raise RuntimeError(
            "Prompt sets differ. "
            f"Missing from model B: {missing_b[0][:160] if missing_b else 'none'}; "
            f"missing from model A: {missing_a[0][:160] if missing_a else 'none'}."
        )
    by_prompt_b = {str(item["prompt"]): item for item in items_b}
    return [(item, by_prompt_b[str(item["prompt"])]) for item in items_a]


def build_html(model_a: Dict[str, object], model_b: Dict[str, object], title: str) -> str:
    """Build the self-contained report HTML, excluding the externally referenced images."""
    pairs = pair_items(model_a, model_b)
    comparisons = []
    for number, (item_a, item_b) in enumerate(pairs, 1):
        delta = abs(float(item_a["score"]) - float(item_b["score"]))
        comparisons.append(
            f"""
        <section class="comparison-row" id="prompt-{number}">
          <div class="prompt-head"><span class="prompt-number">{number:02d}</span>
            <p>{esc(item_a['prompt'])}</p><span class="delta">raw Δ {delta:.2f}</span>
          </div>
          <div class="pair">{image_card(item_a)}{image_card(item_b)}</div>
        </section>"""
        )

    rankings = []
    for model in (model_a, model_b):
        items = model["items"]
        assert isinstance(items, list)
        rows = []
        for rank, item in enumerate(
            sorted(items, key=lambda row: float(row["score"]), reverse=True), 1
        ):
            rows.append(
                f"""
            <article class="rank-item">{image_card(item, rank)}
              <p class="rank-prompt">{esc(item['prompt'])}</p>
            </article>"""
            )
        metric = items[0]["metric"] if items else "score"
        rankings.append(
            f"""
        <section class="model-ranking">
          <div class="ranking-head"><div><span class="eyebrow">MODEL RANKING</span>
            <h3>{esc(model['label'])}</h3></div>
            <span class="pill">{esc(metric)} · step {model['step']}</span>
          </div><div class="rank-grid">{''.join(rows)}</div>
        </section>"""
        )

    step_label = (
        f"{model_a['label']} step {model_a['step']} · "
        f"{model_b['label']} step {model_b['step']}"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}：{esc(model_a['label'])} vs. {esc(model_b['label'])}</title>
<style>
:root{{--ink:#171714;--muted:#6d6b64;--paper:#f5f2eb;--line:#dcd7ca;--accent:#e8552d}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans SC","PingFang SC",system-ui,sans-serif}}
.shell{{width:min(1480px,calc(100% - 40px));margin:auto}}header{{padding:56px 0 34px;border-bottom:1px solid var(--line)}}
.kicker,.eyebrow{{font-size:11px;font-weight:800;letter-spacing:.18em;color:var(--accent)}}h1{{max-width:1150px;margin:12px 0 18px;font:700 clamp(34px,5vw,70px)/1.02 Georgia,serif;letter-spacing:-.04em}}
.lede{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:var(--muted)}}.pill{{border:1px solid var(--line);border-radius:999px;padding:7px 11px;background:#ffffffa8;font-size:12px;color:var(--muted)}}
nav{{position:sticky;top:0;z-index:20;background:#f5f2ebed;backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}}nav .shell{{display:flex;gap:8px;padding:10px 0}}nav a{{padding:9px 14px;border-radius:8px;color:var(--ink);text-decoration:none;font-weight:700;font-size:13px}}
main{{padding:42px 0 80px}}.section-title{{display:flex;justify-content:space-between;align-items:end;gap:18px;margin:0 0 22px}}h2{{font:700 clamp(28px,4vw,48px)/1 Georgia,serif;margin:7px 0}}.section-note{{max-width:550px;color:var(--muted);font-size:13px;line-height:1.6}}
#comparison{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:28px}}#comparison>.section-title{{grid-column:1/-1}}
.comparison-row{{min-width:0;padding:20px 0 30px;border-top:1px solid var(--line);scroll-margin-top:60px}}.prompt-head{{display:grid;grid-template-columns:38px minmax(0,1fr);gap:8px 12px;align-items:start;margin-bottom:14px}}
.prompt-number{{font:700 24px/1 Georgia,serif;color:#aaa59a}}.prompt-head p{{height:7.75em;overflow:auto;margin:0;line-height:1.55;font-size:14px;padding-right:4px}}.delta{{grid-column:2;justify-self:start;font-size:11px;color:var(--muted);background:#fff9;padding:6px 9px;border-radius:999px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.image-card{{min-width:0}}.image-wrap{{position:relative;overflow:hidden;border-radius:10px;background:#dedbd3}}.image-wrap img{{display:block;width:100%;height:auto;max-height:720px;object-fit:contain;background:#e7e3da;transition:transform .25s ease}}.image-wrap:hover img{{transform:scale(1.012)}}
.card-meta{{display:flex;align-items:center;padding:10px 2px 0;font-size:13px}}.dot{{width:9px;height:9px;border-radius:50%;background:var(--dot);margin-right:7px}}.model-name{{font-weight:700}}.score{{margin-left:auto;font:700 18px/1 ui-monospace,monospace}}
#ranking{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:28px;padding-top:74px}}#ranking>.section-title{{grid-column:1/-1}}.model-ranking{{min-width:0;margin-top:30px;padding-top:26px;border-top:2px solid var(--ink)}}
.ranking-head{{display:flex;justify-content:space-between;align-items:end;gap:16px;margin-bottom:20px}}h3{{margin:5px 0 0;font:700 30px/1.1 Georgia,serif}}.rank-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:26px 12px}}
.rank-badge{{position:absolute;z-index:2;top:10px;left:10px;color:#fff;background:#171714e8;border-radius:999px;padding:7px 10px;font-weight:800;font-size:12px}}.rank-prompt{{font-size:12px;line-height:1.55;color:var(--muted);margin:9px 2px 0;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}}
footer{{border-top:1px solid var(--line);padding:24px 0 40px;color:var(--muted);font-size:12px}}
@media(max-width:1100px){{.rank-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:780px){{.shell{{width:min(100% - 24px,1480px)}}header{{padding-top:36px}}#comparison,#ranking{{grid-template-columns:1fr}}.prompt-head p{{height:auto;max-height:9.3em}}.rank-grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}.ranking-head,.section-title{{align-items:start;flex-direction:column}}}}
@media(max-width:480px){{.pair,.rank-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div class="shell"><div class="kicker">FLOW-FACTORY / LATEST EVAL</div>
<h1>{esc(title)}<br>{esc(model_a['label'])} vs. {esc(model_b['label'])}</h1>
<div class="lede"><span class="pill">{esc(step_label)}</span><span class="pill">{len(pairs)} shared prompts</span></div></div></header>
<nav><div class="shell"><a href="#comparison">01 · 同 Prompt 对比</a><a href="#ranking">02 · 模型内排名</a></div></nav>
<main class="shell"><section id="comparison"><div class="section-title"><div><span class="eyebrow">PART ONE</span><h2>同 Prompt 对比</h2></div><p class="section-note">每组使用相同 prompt；分数来自各 run 的最新完整 eval。raw Δ 仅表示原始数值差，不代表跨 reward 标定后的差异。</p></div>{''.join(comparisons)}</section>
<section id="ranking"><div class="section-title"><div><span class="eyebrow">PART TWO</span><h2>模型内排名</h2></div><p class="section-note">两个模型分别按各自 eval reward 从高到低排序。</p></div>{''.join(rankings)}</section></main>
<footer><div class="shell">Static report · original W&amp;B media · no image copies</div></footer></body></html>"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--label-a", default="Model A")
    parser.add_argument("--label-b", default="Model B")
    parser.add_argument("--color-a", default="#ff6b35")
    parser.add_argument("--color-b", default="#4e7cff")
    parser.add_argument("--title", default="Flow-Factory latest eval comparison")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--preview-base-url", default="")
    return parser.parse_args()


def result_summary(model: Dict[str, object]) -> Dict[str, object]:
    """Create the validation summary printed after generation."""
    items = model["items"]
    assert isinstance(items, list)
    missing = sum(not Path(str(item["url"])).is_file() for item in items)
    return {
        "label": model["label"],
        "wandb_file": str(model["wandb_file"]),
        "image_key": model["image_key"],
        "step": model["step"],
        "count": len(items),
        "missing_images": missing,
    }


def main() -> None:
    """Generate and validate the comparison report."""
    args = parse_args()
    model_a = load_model(args.run_a, args.label_a, args.color_a, args.image_key)
    model_b = load_model(args.run_b, args.label_b, args.color_b, args.image_key)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(model_a, model_b, args.title), encoding="utf-8")
    result = {
        "output": str(output),
        "model_a": result_summary(model_a),
        "model_b": result_summary(model_b),
    }
    if args.preview_base_url:
        result["share_url"] = args.preview_base_url.rstrip("/") + quote(str(output), safe="/")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
