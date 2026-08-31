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

"""Shared JSONL batch-inference utilities."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from peft import PeftModel


@dataclass(frozen=True)
class PromptRecord:
    """Represent one normalized prompt selected from a JSONL file."""

    index: int
    line_number: int
    prompt: str


def normalize_prompt(value: object, *, source: Path, line_number: int, raw: bool) -> str:
    """Normalize a plain prompt or extract a nested comprehensive caption.

    Args:
        value: Raw prompt value from the JSONL record.
        source: Source JSONL path used for error context.
        line_number: One-based source line number.
        raw: Whether to keep a JSON-encoded prompt string unchanged.

    Returns:
        Plain prompt text to send to the diffusion pipeline.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{source}:{line_number}: prompt must be a string, got {type(value).__name__}"
        )
    if not value.strip():
        raise ValueError(f"{source}:{line_number}: prompt must not be empty")
    if raw:
        return value

    try:
        nested = json.loads(value)
    except json.JSONDecodeError:
        return value.strip()

    if not isinstance(nested, dict):
        return value.strip()
    caption = nested.get("comprehensive_t2i_caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(
            f"{source}:{line_number}: nested prompt JSON does not contain a nonempty "
            "'comprehensive_t2i_caption'; pass --raw_prompt to use it unchanged"
        )
    return caption.strip()


def load_prompt_records(
    prompt_file: Path,
    *,
    prompt_key: str,
    raw_prompt: bool,
    start_index: int,
    max_samples: Optional[int],
) -> List[PromptRecord]:
    """Load and normalize a selected range of JSONL prompt records.

    Args:
        prompt_file: Input JSONL file.
        prompt_key: Record field containing prompt text.
        raw_prompt: Whether to skip nested caption extraction.
        start_index: Zero-based nonempty-record index to start from.
        max_samples: Maximum selected record count, or ``None`` for all.

    Returns:
        Selected prompt records with stable source indices.
    """
    if not prompt_file.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_file}")
    if start_index < 0:
        raise ValueError("--start_index must be nonnegative")
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max_samples must be a positive integer")

    records: List[PromptRecord] = []
    record_index = 0
    with prompt_file.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{prompt_file}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{prompt_file}:{line_number}: record must be a JSON object")
            if prompt_key not in record:
                raise ValueError(f"{prompt_file}:{line_number}: record has no {prompt_key!r} field")

            if record_index >= start_index:
                prompt = normalize_prompt(
                    record[prompt_key],
                    source=prompt_file,
                    line_number=line_number,
                    raw=raw_prompt,
                )
                records.append(
                    PromptRecord(index=record_index, line_number=line_number, prompt=prompt)
                )
                if max_samples is not None and len(records) >= max_samples:
                    break
            record_index += 1

    if not records:
        raise ValueError(
            f"No prompts selected from {prompt_file} with start_index={start_index} "
            f"and max_samples={max_samples}"
        )
    return records


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model_path: str,
    default_num_inference_steps: int,
) -> None:
    """Add the shared batch-inference CLI arguments.

    Args:
        parser: Parser to extend.
        default_model_path: Default base-model directory.
        default_num_inference_steps: Model-specific denoising step count.
    """
    parser.add_argument("--prompt_file", "--prompt-file", type=Path, required=True)
    parser.add_argument("--lora_path", "--lora-path", type=Path)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--model_path", "--model-path", default=default_model_path)
    parser.add_argument("--prompt_key", "--prompt-key", default="prompt")
    parser.add_argument("--raw_prompt", "--raw-prompt", action="store_true")
    parser.add_argument("--start_index", "--start-index", type=int, default=0)
    parser.add_argument("--max_samples", "--max-samples", type=int)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--num_inference_steps",
        "--num-inference-steps",
        type=int,
        default=default_num_inference_steps,
    )
    parser.add_argument("--guidance_scale", "--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cpu_offload", "--no-cpu-offload", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def attach_lora(pipe: Any, lora_path: Optional[Path]) -> bool:
    """Attach a PEFT LoRA to a pipeline transformer when requested.

    Args:
        pipe: Diffusers pipeline exposing a ``transformer`` component.
        lora_path: PEFT checkpoint directory, or ``None`` for base inference.

    Returns:
        Whether a LoRA was attached.
    """
    if lora_path is None:
        return False
    if not lora_path.is_dir():
        raise FileNotFoundError(f"LoRA checkpoint directory does not exist: {lora_path}")

    config_path = lora_path / "adapter_config.json"
    weight_candidates = [
        lora_path / "adapter_model.safetensors",
        lora_path / "adapter_model.bin",
    ]
    if not config_path.is_file():
        raise FileNotFoundError(f"LoRA adapter config does not exist: {config_path}")
    weight_path = next((path for path in weight_candidates if path.is_file()), None)
    if weight_path is None:
        raise FileNotFoundError(
            f"LoRA checkpoint has no adapter_model.safetensors or adapter_model.bin: {lora_path}"
        )
    try:
        with weight_path.open("rb"):
            pass
    except OSError as error:
        raise PermissionError(f"LoRA weights are not readable: {weight_path}: {error}") from error

    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        str(lora_path),
        torch_dtype=torch.bfloat16,
    )
    return True


def validate_output_targets(
    output_dir: Path,
    records: List[PromptRecord],
    *,
    overwrite: bool,
) -> Path:
    """Create the output directory and reject accidental overwrites.

    Args:
        output_dir: Destination directory.
        records: Selected records defining deterministic image names.
        overwrite: Whether existing outputs may be replaced.

    Returns:
        Metadata JSONL destination.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    if overwrite:
        return metadata_path

    existing = [output_dir / f"{record.index:06d}.png" for record in records]
    existing = [path for path in existing if path.exists()]
    if metadata_path.exists() or existing:
        examples = [str(path) for path in existing[:3]]
        if metadata_path.exists():
            examples.insert(0, str(metadata_path))
        raise FileExistsError(
            "Output targets already exist; pass --overwrite to replace them: " + ", ".join(examples)
        )
    return metadata_path


def write_batch_outputs(
    records: List[PromptRecord],
    *,
    args: argparse.Namespace,
    model_name: str,
    generate_image: Callable[[str, int], Any],
    metadata_path: Path,
) -> None:
    """Generate, save, and journal one image per prompt record.

    Args:
        records: Selected prompt records.
        args: Shared parsed CLI arguments.
        model_name: Human-readable pipeline identifier.
        generate_image: Callable accepting prompt and the configured seed.
        metadata_path: Destination metadata JSONL file.
    """
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for position, record in enumerate(records, start=1):
            image_seed = args.seed
            image = generate_image(record.prompt, image_seed)
            image_name = f"{record.index:06d}.png"
            image_path = args.output_dir / image_name
            image.save(image_path)

            metadata: Dict[str, Any] = {
                "index": record.index,
                "source_line": record.line_number,
                "prompt": record.prompt,
                "seed": image_seed,
                "image": image_name,
                "model": model_name,
                "model_path": str(args.model_path),
                "lora_path": str(args.lora_path) if args.lora_path is not None else None,
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
            }
            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            metadata_file.flush()
            print(f"[{position}/{len(records)}] saved {image_path}")

    print(f"Saved metadata to {metadata_path}")
