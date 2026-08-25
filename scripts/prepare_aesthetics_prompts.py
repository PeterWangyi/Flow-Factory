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

"""Prepare compact prompt JSONL files from the aesthetics dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def normalize_prompt(value: object, *, source: Path, line_number: int) -> str:
    """Extract the comprehensive caption from a nested prompt when present.

    Args:
        value: Raw prompt value from the JSONL record.
        source: Source file used for error context.
        line_number: One-based source line number used for error context.

    Returns:
        The extracted comprehensive caption, or the original plain prompt.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{source}:{line_number}: 'prompt' must be a string, got " f"{type(value).__name__}"
        )

    try:
        nested = json.loads(value)
    except json.JSONDecodeError:
        return value

    if not isinstance(nested, dict):
        return value

    caption = nested.get("comprehensive_t2i_caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(
            f"{source}:{line_number}: nested prompt JSON does not contain a "
            "nonempty 'comprehensive_t2i_caption'"
        )
    return caption.strip()


def iter_records(source: Path) -> Iterator[Dict[str, Any]]:
    """Yield normalized records from one JSONL split.

    Args:
        source: Input JSONL split.

    Yields:
        Records with a normalized plain-text prompt.
    """
    with source.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: record must be a JSON object")
            if "prompt" not in record:
                raise ValueError(f"{source}:{line_number}: record has no 'prompt' field")

            record["prompt"] = normalize_prompt(
                record["prompt"], source=source, line_number=line_number
            )
            yield record


def select_records(
    source: Path, limit: Optional[int], random_generator: random.Random
) -> Iterator[Dict[str, Any]]:
    """Select a deterministic uniform reservoir when a split limit is set.

    Args:
        source: Input JSONL split.
        limit: Maximum number of records, or ``None`` to retain the full split.
        random_generator: Seeded random generator for reservoir sampling and shuffling.

    Yields:
        Normalized records in deterministic sampled order.
    """
    if limit is None:
        yield from iter_records(source)
        return
    if limit < 1:
        raise ValueError("split limits must be positive integers")

    reservoir: List[Dict[str, Any]] = []
    for index, record in enumerate(iter_records(source)):
        if index < limit:
            reservoir.append(record)
            continue
        replacement = random_generator.randint(0, index)
        if replacement < limit:
            reservoir[replacement] = record

    random_generator.shuffle(reservoir)
    yield from reservoir


def write_split(
    source: Path,
    destination: Path,
    *,
    limit: Optional[int],
    seed: int,
    overwrite: bool,
) -> int:
    """Normalize one split and atomically publish the destination JSONL.

    Args:
        source: Input JSONL split.
        destination: Output JSONL split.
        limit: Maximum number of output records, or ``None`` for the full split.
        seed: Random seed used for sampling.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        Number of records written.
    """
    if not source.is_file():
        raise FileNotFoundError(f"Dataset split does not exist: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination}. Pass --overwrite to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            records = select_records(source, limit, random.Random(seed))
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return count


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    train_limit: Optional[int],
    test_limit: Optional[int],
    seed: int,
    overwrite: bool,
) -> Tuple[int, int]:
    """Prepare train and test splits without modifying the source dataset.

    Args:
        input_dir: Directory containing ``train.jsonl`` and ``test.jsonl``.
        output_dir: Separate directory for the normalized dataset.
        train_limit: Maximum training records, or ``None`` for the full split.
        test_limit: Maximum evaluation records, or ``None`` for the full split.
        seed: Random seed used for deterministic sampling.
        overwrite: Whether existing output split files may be replaced.

    Returns:
        A pair containing the written train and test record counts.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")

    train_count = write_split(
        input_dir / "train.jsonl",
        output_dir / "train.jsonl",
        limit=train_limit,
        seed=seed,
        overwrite=overwrite,
    )
    test_count = write_split(
        input_dir / "test.jsonl",
        output_dir / "test.jsonl",
        limit=test_limit,
        seed=seed + 1,
        overwrite=overwrite,
    )
    return train_count, test_count


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract comprehensive_t2i_caption from nested aesthetics prompts and "
            "write a separate Flow-Factory dataset."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Prepare the requested dataset and report the output counts."""
    args = parse_args()
    train_count, test_count = prepare_dataset(
        args.input_dir,
        args.output_dir,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Prepared {train_count} train and {test_count} test prompts in {args.output_dir}")


if __name__ == "__main__":
    main()
