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

"""Launch JSONL batch inference from one YAML configuration."""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_ROOT = Path(__file__).resolve().parent / "workers"

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "flux2-klein": {
        "worker": "inference_data_flux2_klein.py",
        "num_steps": 28,
        "negative_prompt": None,
    },
    "qwen-image": {
        "worker": "inference_data_qwen_image.py",
        "num_steps": 50,
        "negative_prompt": " ",
    },
    "z-image": {
        "worker": "inference_data_zimage.py",
        "num_steps": 40,
        "negative_prompt": "",
    },
}

TOP_LEVEL_KEYS = {
    "version",
    "model",
    "data",
    "output_dir",
    "launcher",
    "sampling",
    "overwrite",
}
MODEL_KEYS = {"type", "path", "checkpoint"}
DATA_KEYS = {"input", "prompt_key", "raw_prompt", "start_index", "limit"}
LAUNCHER_KEYS = {"gpus", "cpu_offload"}
SAMPLING_KEYS = {
    "resolution",
    "num_steps",
    "guidance_scale",
    "seed",
    "negative_prompt",
    "num_images_per_prompt",
}


class RolloutConfigError(ValueError):
    """Represent an invalid rollout configuration."""


class UniqueKeyLoader(yaml.SafeLoader):
    """Load YAML while rejecting duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> Dict[Any, Any]:
    """Construct one YAML mapping and reject duplicate keys."""
    loader.flatten_mapping(node)
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse rollout launcher arguments.

    Args:
        argv: Optional argument list for tests or programmatic use.

    Returns:
        Parsed launcher arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Rollout YAML path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted YAML value, for example --set sampling.num_steps=30.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and command without loading a model.",
    )
    return parser.parse_args(argv)


def _repo_path(value: Union[str, Path]) -> Path:
    """Resolve one local path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load one rollout YAML file."""
    resolved = _repo_path(path)
    if not resolved.is_file():
        raise RolloutConfigError(f"rollout config does not exist: {resolved}")
    try:
        loaded = yaml.load(resolved.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise RolloutConfigError(f"cannot read rollout config {resolved}: {error}") from error
    if not isinstance(loaded, dict):
        raise RolloutConfigError("rollout config root must be a YAML mapping")
    return loaded


def _parse_override(text: str) -> Tuple[List[str], Any]:
    """Parse one dotted ``--set`` override."""
    if "=" not in text:
        raise RolloutConfigError(f"invalid --set {text!r}; expected KEY=VALUE")
    dotted_key, raw_value = text.split("=", 1)
    keys = dotted_key.split(".")
    if not dotted_key or any(not key for key in keys):
        raise RolloutConfigError(f"invalid --set key: {dotted_key!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise RolloutConfigError(f"invalid YAML value in --set {text!r}: {error}") from error
    return keys, value


def _apply_overrides(job: Dict[str, Any], overrides: List[str]) -> None:
    """Apply command-line overrides to a loaded YAML mapping."""
    for override in overrides:
        keys, value = _parse_override(override)
        current = job
        for key in keys[:-1]:
            child = current.get(key)
            if child is None:
                child = {}
                current[key] = child
            if not isinstance(child, dict):
                raise RolloutConfigError(
                    f"cannot apply --set {override!r}: {key!r} is not a mapping"
                )
            current = child
        current[keys[-1]] = value


def _mapping(
    parent: Dict[str, Any],
    name: str,
    allowed: set,
    *,
    required: bool = False,
) -> Dict[str, Any]:
    """Read and validate one nested mapping."""
    value = parent.get(name)
    if value is None:
        if required:
            raise RolloutConfigError(f"missing required mapping: {name}")
        return {}
    if not isinstance(value, dict):
        raise RolloutConfigError(f"{name} must be a mapping")
    non_string = [key for key in value if not isinstance(key, str)]
    if non_string:
        raise RolloutConfigError(f"{name} contains non-string keys: {non_string!r}")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RolloutConfigError(f"unknown {name} keys: {', '.join(unknown)}")
    return value


def _required_string(mapping: Dict[str, Any], key: str, label: str) -> str:
    """Read one required nonempty string."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RolloutConfigError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: Any, label: str) -> Optional[str]:
    """Read one optional nonempty string."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RolloutConfigError(f"{label} must be a non-empty string or null")
    return value


def _integer(value: Any, label: str, *, minimum: Optional[int] = None) -> int:
    """Validate one integer."""
    if type(value) is not int:
        raise RolloutConfigError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise RolloutConfigError(f"{label} must be >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: Optional[float] = None) -> float:
    """Validate one finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RolloutConfigError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise RolloutConfigError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise RolloutConfigError(f"{label} must be >= {minimum}")
    return result


def _resolution(value: Any) -> List[int]:
    """Normalize a square or ``H x W`` resolution."""
    if type(value) is int:
        size = _integer(value, "sampling.resolution", minimum=1)
        return [size, size]
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
        if match:
            return [int(match.group(1)), int(match.group(2))]
        if value.strip().isdigit():
            size = int(value.strip())
            return [size, size]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [
            _integer(item, f"sampling.resolution[{index}]", minimum=1)
            for index, item in enumerate(value)
        ]
    raise RolloutConfigError(
        "sampling.resolution must be an integer, [H, W], or a string like 1024x1024"
    )


def _gpus(value: Any) -> List[int]:
    """Normalize the configured GPU selection."""
    if value is None:
        items = [0]
    elif type(value) is int:
        items = [value]
    elif isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise RolloutConfigError("launcher.gpus must be a GPU index, comma list, or list")
    # Preserve the configured order: it is used both for CUDA visibility and for
    # mapping worker-local ranks back to the physical GPU identifiers.
    result: List[int] = []
    for index, item in enumerate(items):
        if isinstance(item, str) and item.isdigit():
            item = int(item)
        result.append(_integer(item, f"launcher.gpus[{index}]", minimum=0))
    if not result:
        raise RolloutConfigError("launcher.gpus must select at least one GPU")
    if len(set(result)) != len(result):
        raise RolloutConfigError("launcher.gpus must not contain duplicate GPU indices")
    return result


def _validate_and_resolve(job: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a rollout job and fill model-specific defaults."""
    non_string = [key for key in job if not isinstance(key, str)]
    if non_string:
        raise RolloutConfigError(f"top level contains non-string keys: {non_string!r}")
    unknown = sorted(set(job) - TOP_LEVEL_KEYS)
    if unknown:
        raise RolloutConfigError(f"unknown top-level keys: {', '.join(unknown)}")

    version = job.get("version", 1)
    if type(version) is not int or version != 1:
        raise RolloutConfigError("version must be integer 1")

    model_raw = _mapping(job, "model", MODEL_KEYS, required=True)
    model_type = _required_string(model_raw, "type", "model.type").lower()
    if model_type not in MODEL_SPECS:
        raise RolloutConfigError(f"model.type must be one of: {', '.join(sorted(MODEL_SPECS))}")
    model_path = _required_string(model_raw, "path", "model.path")
    checkpoint = _optional_string(model_raw.get("checkpoint"), "model.checkpoint")

    data_raw = _mapping(job, "data", DATA_KEYS, required=True)
    input_path = _repo_path(_required_string(data_raw, "input", "data.input"))
    if not input_path.is_file():
        raise RolloutConfigError(f"data.input does not exist: {input_path}")
    prompt_key = data_raw.get("prompt_key", "prompt")
    if not isinstance(prompt_key, str) or not prompt_key.strip():
        raise RolloutConfigError("data.prompt_key must be a non-empty string")
    raw_prompt = data_raw.get("raw_prompt", False)
    if not isinstance(raw_prompt, bool):
        raise RolloutConfigError("data.raw_prompt must be a boolean")
    start_index = _integer(data_raw.get("start_index", 0), "data.start_index", minimum=0)
    limit = data_raw.get("limit")
    if limit is not None:
        limit = _integer(limit, "data.limit", minimum=1)

    output_dir = _repo_path(_required_string(job, "output_dir", "output_dir"))
    overwrite = job.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise RolloutConfigError("overwrite must be a boolean")

    launcher_raw = _mapping(job, "launcher", LAUNCHER_KEYS)
    gpus = _gpus(launcher_raw.get("gpus"))
    cpu_offload = launcher_raw.get("cpu_offload", True)
    if not isinstance(cpu_offload, bool):
        raise RolloutConfigError("launcher.cpu_offload must be a boolean")

    sampling_raw = _mapping(job, "sampling", SAMPLING_KEYS)
    spec = MODEL_SPECS[model_type]
    resolution = _resolution(sampling_raw.get("resolution", 1024))
    num_steps = _integer(
        sampling_raw.get("num_steps", spec["num_steps"]),
        "sampling.num_steps",
        minimum=1,
    )
    guidance_scale = _number(
        sampling_raw.get("guidance_scale", 4.0),
        "sampling.guidance_scale",
        minimum=0,
    )
    seed = _integer(sampling_raw.get("seed", 42), "sampling.seed", minimum=0)
    # This count expands every prompt into independent sample tasks. The worker
    # later assigns those tasks by position, so the seed remains rank-independent.
    num_images_per_prompt = _integer(
        sampling_raw.get("num_images_per_prompt", 1),
        "sampling.num_images_per_prompt",
        minimum=1,
    )
    negative_prompt = sampling_raw.get("negative_prompt", spec["negative_prompt"])
    if model_type == "flux2-klein" and negative_prompt is not None:
        raise RolloutConfigError("sampling.negative_prompt is not supported by flux2-klein")
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise RolloutConfigError("sampling.negative_prompt must be a string or null")

    return {
        "version": 1,
        "model": {
            "type": model_type,
            "path": model_path,
            "checkpoint": checkpoint,
        },
        "data": {
            "input": str(input_path),
            "prompt_key": prompt_key,
            "raw_prompt": raw_prompt,
            "start_index": start_index,
            "limit": limit,
        },
        "output_dir": str(output_dir),
        "launcher": {"gpus": gpus, "cpu_offload": cpu_offload},
        "sampling": {
            "resolution": resolution,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "negative_prompt": negative_prompt,
            "num_images_per_prompt": num_images_per_prompt,
        },
        "overwrite": overwrite,
    }


def _command(config: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    """Build the selected batch worker command and environment."""
    model = config["model"]
    data = config["data"]
    sampling = config["sampling"]
    height, width = sampling["resolution"]
    worker = WORKER_ROOT / MODEL_SPECS[model["type"]]["worker"]
    # The worker receives the complete GPU set and creates one full model replica
    # per rank; no distributed process group or model-parallel state is required.
    command = [
        sys.executable,
        str(worker),
        "--prompt-file",
        data["input"],
        "--output-dir",
        config["output_dir"],
        "--model-path",
        model["path"],
        "--prompt-key",
        data["prompt_key"],
        "--start-index",
        str(data["start_index"]),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num-inference-steps",
        str(sampling["num_steps"]),
        "--guidance-scale",
        str(sampling["guidance_scale"]),
        "--seed",
        str(sampling["seed"]),
        "--num-gpus",
        str(len(config["launcher"]["gpus"])),
        "--gpu-ids",
        ",".join(str(gpu) for gpu in config["launcher"]["gpus"]),
        "--num-images-per-prompt",
        str(sampling["num_images_per_prompt"]),
    ]
    if model["checkpoint"] is not None:
        command.extend(["--lora-path", str(_repo_path(model["checkpoint"]))])
    if data["raw_prompt"]:
        command.append("--raw-prompt")
    if data["limit"] is not None:
        command.extend(["--max-samples", str(data["limit"])])
    if not config["launcher"]["cpu_offload"]:
        command.append("--no-cpu-offload")
    if config["overwrite"]:
        command.append("--overwrite")
    if sampling["negative_prompt"] is not None:
        command.extend(["--negative-prompt", sampling["negative_prompt"]])

    env = os.environ.copy()
    # CUDA_VISIBLE_DEVICES remaps the selected physical devices to local ordinals
    # 0..N-1, which lets each spawned rank select its own visible GPU deterministically.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in config["launcher"]["gpus"])
    return command, env


def _display_command(command: List[str], config: Dict[str, Any]) -> str:
    """Render one copyable shell command."""
    visible_gpu = ",".join(str(gpu) for gpu in config["launcher"]["gpus"])
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(visible_gpu)} {shlex.join(command)}"


def main(argv: Optional[List[str]] = None) -> int:
    """Validate a YAML job and launch its selected inference worker.

    Args:
        argv: Optional argument list for tests or programmatic use.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    try:
        job = _load_yaml(args.config)
        _apply_overrides(job, args.overrides)
        config = _validate_and_resolve(job)
        command, env = _command(config)
    except RolloutConfigError as error:
        print(f"rollout config error: {error}", file=sys.stderr)
        return 2

    print("Resolved rollout config:")
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True).rstrip())
    print("\nLaunch command:")
    print(_display_command(command, config), flush=True)
    if args.dry_run:
        return 0

    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    except FileNotFoundError as error:
        print(f"cannot launch rollout: {error}", file=sys.stderr)
        return 127
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())


# cd /mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory && conda activate /mnt/aigc/wangyubo/anaconda3/envs/flowfactory/

# python \
#   tools/rollout/rollout.py \
#   --config tools/rollout/peter_config/z_image.yaml
