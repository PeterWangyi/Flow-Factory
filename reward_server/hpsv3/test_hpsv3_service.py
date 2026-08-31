#!/usr/bin/env python3
"""Smoke-test the HPSv3 HTTP reward service with one image."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_URL = "http://10.119.26.83:9010"
DEFAULT_IMAGE = (
    "/mnt/aigc/wangyubo/data/IG/eval_results/july_test/0723_test/u15/2k/U15_offical_mt50k/eval_full_20260731/images/qwenimagebench/0001_0.png"
)
DEFAULT_PROMPT = "test"


def normalize_base_url(url: str) -> str:
    return url.rstrip("/") + "/"


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return args.prompt


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, Any, float]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            elapsed = time.perf_counter() - started
            try:
                data = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                data = raw
            return resp.status, data, elapsed
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        try:
            data = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            data = raw
        return exc.code, data, elapsed


def check_health(base_url: str, timeout: float) -> bool:
    for endpoint in ("healthz", "health"):
        url = urljoin(base_url, endpoint)
        try:
            status, data, elapsed = request_json("GET", url, timeout=timeout)
        except URLError as exc:
            print(f"[health] GET /{endpoint} failed: {exc}", file=sys.stderr)
            continue

        print(f"[health] GET /{endpoint} -> HTTP {status} ({elapsed:.2f}s)")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if 200 <= status < 300:
            return True

    return False


def encode_image(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def score_image(base_url: str, image_path: Path, prompt: str, timeout: float) -> int:
    payload = {
        "prompt": prompt,
        "image_base64": encode_image(image_path),
    }
    url = urljoin(base_url, "score")
    status, data, elapsed = request_json("POST", url, payload=payload, timeout=timeout)

    print(f"[score] POST /score -> HTTP {status} ({elapsed:.2f}s)")
    print(f"[score] image: {image_path}")
    print(f"[score] prompt chars: {len(prompt)}")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    if not (200 <= status < 300):
        return 1
    if not isinstance(data, dict) or "score" not in data:
        print("[score] response does not contain a 'score' field", file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether an HPSv3 service is reachable and can score one image."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"default: {DEFAULT_URL}")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="image path to score")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"prompt paired with the image; default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="read prompt text from a UTF-8 file; overrides --prompt",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="skip /healthz and /health checks and call /score directly",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = normalize_base_url(args.url)
    image_path = Path(args.image)
    prompt = read_prompt(args)

    print(f"[config] url: {base_url.rstrip('/')}")
    print(f"[config] timeout: {args.timeout}s")

    try:
        if not args.skip_health:
            healthy = check_health(base_url, timeout=args.timeout)
            if not healthy:
                print("[health] no health endpoint returned HTTP 2xx; trying /score anyway")
        return score_image(base_url, image_path, prompt, timeout=args.timeout)
    except (FileNotFoundError, URLError, TimeoutError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# /mnt/aigc/wangyubo/data/IG/eval_results/apr_test/0416_test/zoe_sft_neo/zoe_sft_Neo_d20260430_Neo_mt84k_info_mt44kema_train_full_info6_und2_t2i15_oth05_seq20480_gpu480_21000steps/eval_20260503/images/bench_ss_infographic_200
# /mnt/aigc/wangyubo/data/IG/eval_results/june_test/0609_test/RL/DPO_21k_base_hpsv3_cn2en8_DPO_1905steps/eval_20260609/images/bench_ss_infographic_200
# /mnt/aigc/wangyubo/data/IG/eval_results/june_test/0609_test/RL/DPO_21k_base_hpsv3_cn2en8_DPO_3810steps/eval_20260609/images/bench_ss_infographic_200
# /mnt/aigc/wangyubo/data/IG/eval_results/apr_test/0416_test/zoe_sft_neo/zoe_sft_Neo_neo9Bpp_info-pickarocrvlm-grpofast-step10_5_3-noiselevel0.7-hpsv3ocrpnt-bgvlmocr-dyres-warmup200-lr1e-5-kl0.01-64GPU-freeze39-cfg4_3_0-sft9k-int1-10_0.1-unify-res1800ocr-s4800_2026.05.09_20.54.07_1200steps/eval_20260510/images/bench_ss_infographic_200
