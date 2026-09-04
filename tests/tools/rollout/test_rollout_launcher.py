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

"""Tests for the YAML rollout launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ROLLOUT_PATH = REPO_ROOT / "tools" / "rollout" / "rollout.py"


def _load_rollout_module():
    spec = importlib.util.spec_from_file_location("rollout_launcher", ROLLOUT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(tmp_path: Path) -> dict:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"prompt": "one"}\n', encoding="utf-8")
    return {
        "version": 1,
        "model": {"type": "z-image", "path": "/models/z-image"},
        "data": {"input": str(prompt_file)},
        "output_dir": str(tmp_path / "outputs"),
        "launcher": {"gpus": [3, 1], "cpu_offload": True},
        "sampling": {"num_images_per_prompt": 4},
    }


def test_gpus_preserve_multiple_device_order() -> None:
    rollout = _load_rollout_module()

    assert rollout._gpus("3,1,5") == [3, 1, 5]


def test_gpus_reject_duplicate_devices() -> None:
    rollout = _load_rollout_module()

    with pytest.raises(rollout.RolloutConfigError, match="duplicate"):
        rollout._gpus([2, 2])


def test_resolved_config_defaults_to_one_image_per_prompt(tmp_path: Path) -> None:
    rollout = _load_rollout_module()
    job = _job(tmp_path)
    job["sampling"].pop("num_images_per_prompt")

    config = rollout._validate_and_resolve(job)

    assert config["sampling"]["num_images_per_prompt"] == 1


def test_command_exposes_all_gpus_and_forwards_image_count(tmp_path: Path) -> None:
    rollout = _load_rollout_module()
    config = rollout._validate_and_resolve(_job(tmp_path))

    command, env = rollout._command(config)

    assert env["CUDA_VISIBLE_DEVICES"] == "3,1"
    assert command[command.index("--num-gpus") + 1] == "2"
    assert command[command.index("--gpu-ids") + 1] == "3,1"
    assert command[command.index("--num-images-per-prompt") + 1] == "4"


def test_single_gpu_command_remains_compatible(tmp_path: Path) -> None:
    rollout = _load_rollout_module()
    job = _job(tmp_path)
    job["launcher"]["gpus"] = [2]
    job["sampling"].pop("num_images_per_prompt")
    config = rollout._validate_and_resolve(job)

    command, env = rollout._command(config)

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert command[command.index("--num-gpus") + 1] == "1"
    assert command[command.index("--gpu-ids") + 1] == "2"
    assert command[command.index("--num-images-per-prompt") + 1] == "1"
