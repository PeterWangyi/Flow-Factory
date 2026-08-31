#!/usr/bin/env python3
"""Resolve a Zhiqian model tag to an instruction constant."""

from __future__ import annotations

import argparse
import importlib
import sys
from types import ModuleType


INSTRUCTION_SUFFIX = "_INSTRUCTION"


def discover_instructions(module: ModuleType) -> dict[str, str]:
    instructions: dict[str, str] = {}
    for name, value in vars(module).items():
        if not name.endswith(INSTRUCTION_SUFFIX):
            continue
        dimension = name[: -len(INSTRUCTION_SUFFIX)].lower()
        if dimension and isinstance(value, str) and value.strip():
            instructions[dimension] = value.strip()
    return instructions


def resolve_instruction(module: ModuleType, model_tag: str) -> tuple[str, str]:
    normalized_tag = model_tag.strip().lower()
    instructions = discover_instructions(module)
    matches = [
        dimension
        for dimension in instructions
        if normalized_tag == dimension or normalized_tag.endswith(f"_{dimension}")
    ]
    if not matches:
        available = ", ".join(sorted(instructions)) or "<none>"
        raise ValueError(
            f"no system prompt matches model tag {model_tag!r}; "
            f"available model suffixes: {available}"
        )

    # Prefer the most specific suffix if names overlap in the future.
    dimension = max(matches, key=len)
    return dimension, instructions[dimension]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="full model tag")
    parser.add_argument(
        "--module",
        default="zhiqian_model_sysprompt",
        help="Python module containing *_INSTRUCTION constants",
    )
    args = parser.parse_args()

    try:
        module = importlib.import_module(args.module)
        dimension, _ = resolve_instruction(module, args.tag)
    except (ImportError, ValueError) as exc:
        print(f"System prompt error: {exc}", file=sys.stderr)
        return 2
    print(dimension)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
