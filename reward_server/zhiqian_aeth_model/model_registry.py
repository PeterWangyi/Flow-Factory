#!/usr/bin/env python3
"""Validate, list, and resolve checkpoints from a simple JSON registry."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REQUIRED_KEYS = {"tag", "model_path"}


class RegistryError(ValueError):
    """Raised when the model registry is invalid or cannot be resolved."""


def load_registry(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RegistryError(f"registry does not exist: {path}")

    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{path}: invalid JSON: {exc.msg}") from exc

    if not isinstance(raw, list):
        raise RegistryError(f"{path}: top-level JSON value must be an array")

    models: list[dict[str, str]] = []
    seen_tags: dict[str, int] = {}

    for index, entry in enumerate(raw):
        location = f"{path}: entry {index}"
        if not isinstance(entry, dict):
            raise RegistryError(f"{location}: must be an object")
        if set(entry) != REQUIRED_KEYS:
            raise RegistryError(
                f"{location}: keys must be exactly: tag, model_path"
            )

        tag = entry["tag"]
        model_path = entry["model_path"]
        if not isinstance(tag, str) or not SAFE_TAG.fullmatch(tag):
            raise RegistryError(
                f"{location}: 'tag' must match {SAFE_TAG.pattern!r}"
            )
        if not isinstance(model_path, str) or not model_path or "\n" in model_path:
            raise RegistryError(
                f"{location}: 'model_path' must be a non-empty path string"
            )
        if tag in seen_tags:
            raise RegistryError(
                f"{location}: duplicate tag {tag!r}; "
                f"first declared in entry {seen_tags[tag]}"
            )

        seen_tags[tag] = index
        models.append({"tag": tag, "model_path": model_path})

    return models


def resolve_model(
    models: list[dict[str, str]], tag: str, *, check_path: bool
) -> dict[str, str]:
    if not SAFE_TAG.fullmatch(tag):
        raise RegistryError(f"tag must match {SAFE_TAG.pattern!r}")

    for entry in models:
        if entry["tag"] == tag:
            if check_path and not Path(entry["model_path"]).is_file():
                raise RegistryError(
                    f"model_path does not exist for tag {tag!r}: "
                    f"{entry['model_path']}"
                )
            return entry

    available = ", ".join(entry["tag"] for entry in models) or "<none>"
    raise RegistryError(f"unknown tag {tag!r}; available tags: {available}")


def print_model_list(models: list[dict[str, str]]) -> None:
    if not models:
        print("No models registered.")
        return

    width = max(len("TAG"), *(len(entry["tag"]) for entry in models))
    print(f"{'TAG':<{width}}  MODEL_PATH")
    for entry in models:
        print(f"{entry['tag']:<{width}}  {entry['model_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path, help="models.json path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="validate and list registered models")

    resolve_parser = subparsers.add_parser("resolve", help="resolve one tag")
    resolve_parser.add_argument("--tag", required=True, help="unique model tag")
    resolve_parser.add_argument(
        "--format",
        choices=("json", "shell"),
        default="json",
        help="output format",
    )
    resolve_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="resolve without requiring model_path to exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        models = load_registry(args.registry)
        if args.command == "list":
            print_model_list(models)
            return 0

        entry = resolve_model(
            models,
            args.tag,
            check_path=not args.allow_missing,
        )
        if args.format == "shell":
            print(f"RESOLVED_MODEL_TAG={shlex.quote(entry['tag'])}")
            print(
                "REGISTERED_CHECKPOINT_PATH="
                f"{shlex.quote(entry['model_path'])}"
            )
        else:
            print(json.dumps(entry, ensure_ascii=False, indent=2))
    except RegistryError as exc:
        print(f"Model registry error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
