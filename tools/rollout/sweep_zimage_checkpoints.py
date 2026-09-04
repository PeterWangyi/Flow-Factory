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

"""Submit one multi-GPU Z-Image rollout job per checkpoint."""

from __future__ import annotations

import argparse
import copy
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_CONFIG = (
    REPO_ROOT / "tools" / "rollout" / "configs" / "peter" / "zoe_inference" / "z_image.yaml"
)


OUTPUT_ROOT_ENV = "FLOW_FACTORY_ROLLOUT_OUTPUT_ROOT"
OUTPUT_ROOT = Path.home() / "flow_factory_rollout"


SWEEP_CONFIG_DIR = REPO_ROOT / ".scratch" / "rollout_sweep_configs"

CHECKPOINTS = [
    {
        "checkpoint": (
            "/mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory/saves/z_image_hpsv3_2x8/"
            "[Z-Image]-D-u15-filter04-R-hpsv3-20260828_195751/checkpoints/checkpoint-180"
        ),
        "save_tag": "checkpoint-180",
    },
    {
        "checkpoint": (
            "/mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory/saves/z_image_hpsv3_2x8/"
            "[Z-Image]-D-U15-filter-human-R-hpsv3-20260902_012622/checkpoints/checkpoint-280"
        ),
        "save_tag": "human-hpsv3-checkpoint-280",
    },
    {
        "checkpoint": (
            "/mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory/saves/z_image_hpsv3_2x8/"
            "[Z-Image]-D-U15-filter-human-R-v022-lighting-20260902_012652/"
            "checkpoints/checkpoint-280"
        ),
        "save_tag": "human-lighting-checkpoint-280",
    },
]


JOB_NAME_MAX_LEN = 63
SCO_BIN = Path(os.environ.get("SCO_BIN", str(Path.home() / ".sco" / "bin" / "sco"))).expanduser()


WORKSPACE_NAME = "aigc"


### aec2 name
## default
# AEC2_NAME="m-train-neo2-interleave"
# AEC2_NAME="m-train-neo2"
# AEC2_NAME="neo1-edit"
# AEC2_NAME="neo1-trajectory"
# AEC2_NAME="neo1-agentic"
# AEC2_NAME="umm"
AEC2_NAME = "m-train-neo2-infographic"

# AEC2_NAME = "vigen"
# AEC2_NAME="si"

WS_SPEC = "N6lS.Iu.I80"  # default
# WS_SPEC = "N6lS.Iq.I10"  # vigen si

NPROC = 8


PRIORITY = "HIGHEST"
CONTAINER_IMAGE_URL = "registry.ms-sc-01.maoshanwangtech.com/ccr_2/ulimit-change:20240725-18h11m01s"
STORAGE_MOUNT = (
    "1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,"
    "047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1,"
    "ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs,"
    "c83d08bc-2965-11ef-b8c5-929f74fd8884:/mnt/aigc/,"
    "01998fb1-b876-7b33-82c9-4427517bf536:/mnt/umm"
)
CONDA_SH = Path("/mnt/aigc/wangyubo/anaconda3/etc/profile.d/conda.sh")
CONDA_ENV = Path("/mnt/aigc/wangyubo/anaconda3/envs/flowfactory")


def _runtime_setting(name: str, default: str) -> str:
    """Read one optional runtime override from the environment."""
    return os.environ.get(name, default)


def _safe_tag(value: str) -> str:
    """Return a checkpoint tag safe for paths and ACP job names."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value.strip()).strip("._-")
    if not safe:
        raise ValueError("checkpoint save_tag must contain a path-safe character")
    return safe


def _parse_checkpoint(value: str) -> Dict[str, str]:
    """Parse a PATH or NAME=PATH checkpoint override."""
    if "=" in value:
        name, checkpoint = value.split("=", 1)
        if not name.strip() or not checkpoint.strip():
            raise ValueError(f"invalid --ckpt value: {value!r}")
        return {"checkpoint": checkpoint.strip(), "save_tag": _safe_tag(name)}
    checkpoint = value.strip()
    if not checkpoint:
        raise ValueError("--ckpt path must not be empty")
    return {"checkpoint": checkpoint, "save_tag": _safe_tag(Path(checkpoint).name)}


def build_checkpoint_config(
    base: Dict[str, Any],
    *,
    checkpoint: Dict[str, Any],
    output_root: Path,
    nproc: int,
) -> Dict[str, Any]:
    """Patch one base rollout config for a checkpoint job.

    Args:
        base: Base rollout configuration mapping.
        checkpoint: Mapping containing ``checkpoint`` and optional ``save_tag`` values.
        output_root: Root directory for checkpoint-specific outputs.
        nproc: Number of GPUs requested by the checkpoint job.

    Returns:
        Independent rollout configuration for the checkpoint.
    """
    if nproc < 1:
        raise ValueError("nproc must be a positive integer")
    checkpoint_path = str(checkpoint.get("checkpoint", "") or "").strip()
    if not checkpoint_path:
        raise ValueError("checkpoint entry must define a non-empty checkpoint path")
    save_tag = _safe_tag(str(checkpoint.get("save_tag", "") or Path(checkpoint_path).name))

    config = copy.deepcopy(base)
    model = config.setdefault("model", {})
    if model.get("type") != "z-image":
        raise ValueError("base rollout config must use model.type: z-image")
    model["checkpoint"] = checkpoint_path
    config["output_dir"] = str(output_root / save_tag)
    launcher = config.setdefault("launcher", {})
    launcher["gpus"] = list(range(nproc))
    return config


def build_submit_command(*, save_tag: str, config_path: Path, nproc: int) -> List[str]:
    """Build one ACP job command that directly runs the rollout config.

    Args:
        save_tag: Checkpoint identifier used in the ACP job name.
        config_path: Generated rollout configuration path.
        nproc: Number of GPUs requested by the checkpoint job.

    Returns:
        Argument vector for ``sco acp jobs create``.
    """
    if nproc < 1:
        raise ValueError("nproc must be a positive integer")
    safe_tag = _safe_tag(save_tag)
    conda_sh = _runtime_setting("FLOW_FACTORY_CONDA_SH", str(CONDA_SH))
    conda_env = _runtime_setting("FLOW_FACTORY_CONDA_ENV", str(CONDA_ENV))
    repo_root = _runtime_setting("FLOW_FACTORY_REPO_ROOT", str(REPO_ROOT))
    rollout_command = " ".join(
        [
            "set -euo pipefail;",
            f"source {shlex.quote(conda_sh)};",
            f"conda activate {shlex.quote(conda_env)};",
            f"cd {shlex.quote(repo_root)};",
            "python tools/rollout/rollout.py",
            f"--config {shlex.quote(str(config_path))}",
        ]
    )
    inner_command = f"bash -lc {shlex.quote(rollout_command)}"
    job_name = f"ff-zimage-rollout-{safe_tag}"[:JOB_NAME_MAX_LEN].rstrip("._-")
    sco_bin = _runtime_setting("SCO_BIN", str(SCO_BIN))
    workspace_name = _runtime_setting("FLOW_FACTORY_WORKSPACE_NAME", WORKSPACE_NAME)
    aec2_name = _runtime_setting("FLOW_FACTORY_AEC2_NAME", AEC2_NAME)
    ws_spec = _runtime_setting("FLOW_FACTORY_WS_SPEC", WS_SPEC)
    priority = _runtime_setting("FLOW_FACTORY_PRIORITY", PRIORITY)
    container_image_url = _runtime_setting("FLOW_FACTORY_CONTAINER_IMAGE_URL", CONTAINER_IMAGE_URL)
    storage_mount = _runtime_setting("FLOW_FACTORY_STORAGE_MOUNT", STORAGE_MOUNT)
    return [
        sco_bin,
        "acp",
        "jobs",
        "create",
        f"--workspace-name={workspace_name}",
        f"--aec2-name={aec2_name}",
        f"--job-name={job_name}",
        f"--priority={priority}",
        f"--container-image-url={container_image_url}",
        f"--storage-mount={storage_mount}",
        "--training-framework=pytorch",
        "--worker-nodes=1",
        f"--worker-spec={ws_spec}.{nproc}",
        f"--command={inner_command}",
    ]


def _load_base_config(path: Path) -> Dict[str, Any]:
    """Load one base rollout YAML mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"base rollout config does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("base rollout config root must be a YAML mapping")
    return loaded


def resolve_output_root(value: Optional[Path]) -> Path:
    """Resolve an explicit, environment-provided, or user-local output root.

    Args:
        value: Command-line output root, or ``None`` to use the environment/default.

    Returns:
        Absolute output root path.
    """
    if value is not None:
        return value.expanduser().resolve()
    configured = os.environ.get(OUTPUT_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return OUTPUT_ROOT.expanduser().resolve()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse checkpoint sweep arguments.

    Args:
        argv: Optional argument list for tests or programmatic use.

    Returns:
        Parsed checkpoint sweep arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=f"Output root (default: ${OUTPUT_ROOT_ENV} or ~/flow_factory_rollout).",
    )
    parser.add_argument("--nproc", type=int, default=NPROC)
    parser.add_argument(
        "--ckpt",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Checkpoint override. Replaces CHECKPOINTS when provided.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Generate checkpoint configs and optionally submit their ACP jobs.

    Args:
        argv: Optional argument list for tests or programmatic use.

    Returns:
        Zero after every config is generated and every requested submission succeeds.
    """
    args = parse_args(argv)
    if args.nproc < 1:
        raise ValueError("--nproc must be a positive integer")

    base_config_path = args.base_config.expanduser().resolve()
    output_root = resolve_output_root(args.output_root)
    checkpoints = [_parse_checkpoint(value) for value in args.ckpt] if args.ckpt else CHECKPOINTS
    if not checkpoints:
        raise ValueError("no checkpoints configured")

    base = _load_base_config(base_config_path)
    SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    seen_tags = set()
    for checkpoint in checkpoints:
        checkpoint_path = Path(str(checkpoint["checkpoint"])).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
        save_tag = _safe_tag(str(checkpoint.get("save_tag", checkpoint_path.name)))
        if save_tag in seen_tags:
            raise ValueError(f"duplicate checkpoint save_tag: {save_tag}")
        seen_tags.add(save_tag)

        config = build_checkpoint_config(
            base,
            checkpoint={"checkpoint": str(checkpoint_path), "save_tag": save_tag},
            output_root=output_root,
            nproc=args.nproc,
        )
        config_path = (SWEEP_CONFIG_DIR / f"{save_tag}.yaml").resolve()
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        command = build_submit_command(
            save_tag=save_tag,
            config_path=config_path,
            nproc=args.nproc,
        )
        print(f"[SWEEP] checkpoint={checkpoint_path}")
        print(f"        output={config['output_dir']}")
        print(f"        config={config_path}")
        print("[SUBMIT]", shlex.join(command))
        if not args.dry_run:
            if not SCO_BIN.is_file():
                raise FileNotFoundError(f"sco executable does not exist: {SCO_BIN}")
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
