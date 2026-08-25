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

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "peter_training" / "test_scripts" / "launch_hpsv3_2x8.sh"


def _dry_run(
    *args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "FF_DRY_RUN": "1",
            "NUM_MACHINES": "2",
            "GPUS_PER_NODE": "8",
        }
    )
    if extra_env:
        environment.update(extra_env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("model", "config"),
    (
        ("flux2", "examples/grpo/lora/flux2/hpsv3_2x8.yaml"),
        ("qwen-image", "examples/grpo/lora/qwen_image/hpsv3_2x8.yaml"),
        ("z-image", "examples/grpo/lora/z_image/hpsv3_2x8.yaml"),
        ("sd3.5", "examples/grpo/lora/sd3_5/hpsv3_2x8_medium.yaml"),
    ),
)
def test_launcher_maps_model_to_config(model: str, config: str) -> None:
    result = _dry_run(model, "0", "10.0.0.1")

    assert result.returncode == 0, result.stderr
    assert f"Config:         {config}" in result.stdout
    assert f"Dry-run command: ff-train {config}" in result.stdout


def test_launcher_accepts_environment_only_invocation() -> None:
    result = _dry_run(
        extra_env={
            "FF_MODEL": "z-image",
            "MACHINE_RANK": "1",
            "MASTER_ADDR": "10.0.0.1",
        }
    )

    assert result.returncode == 0, result.stderr
    assert "Model:          z-image" in result.stdout
    assert "Machine rank:   1" in result.stdout


def test_launcher_rejects_invalid_rank() -> None:
    result = _dry_run("qwen-image", "2", "10.0.0.1")

    assert result.returncode == 2
    assert "machine_rank must be 0 or 1" in result.stderr


def test_launcher_rejects_unknown_model() -> None:
    result = _dry_run("unknown", "0", "10.0.0.1")

    assert result.returncode == 2
    assert "Unsupported model" in result.stderr
