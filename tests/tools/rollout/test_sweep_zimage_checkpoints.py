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

"""Tests for one-command Z-Image checkpoint rollout submission."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEP_PATH = REPO_ROOT / "tools" / "rollout" / "sweep_zimage_checkpoints.py"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("sweep_zimage_checkpoints", SWEEP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_config(tmp_path: Path) -> Path:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"prompt": "one"}\n', encoding="utf-8")
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "model": {
                    "type": "z-image",
                    "path": "/models/z-image",
                    "checkpoint": "/old/checkpoint",
                },
                "data": {"input": str(prompt_file)},
                "output_dir": "/old/output",
                "launcher": {"gpus": [0], "cpu_offload": True},
                "sampling": {"num_images_per_prompt": 1},
                "overwrite": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_build_checkpoint_config_isolates_output_and_gpu_list(tmp_path: Path) -> None:
    sweep = _load_sweep_module()
    base = yaml.safe_load(_base_config(tmp_path).read_text(encoding="utf-8"))

    config = sweep.build_checkpoint_config(
        base,
        checkpoint={"checkpoint": "/checkpoints/180", "save_tag": "step-180"},
        output_root=tmp_path / "outputs",
        nproc=4,
    )

    assert config["model"]["checkpoint"] == "/checkpoints/180"
    assert config["output_dir"] == str(tmp_path / "outputs" / "step-180")
    assert config["launcher"]["gpus"] == [0, 1, 2, 3]


def test_submit_command_runs_rollout_directly_without_sleep(tmp_path: Path) -> None:
    sweep = _load_sweep_module()
    config_path = tmp_path / "step-180.yaml"

    command = sweep.build_submit_command(
        save_tag="step-180",
        config_path=config_path,
        nproc=4,
    )
    joined = " ".join(command)
    inner_command = next(item for item in command if item.startswith("--command="))

    assert command[0] == str(Path.home() / ".sco" / "bin" / "sco")
    assert "--worker-spec=N6lS.Iu.I80.4" in command
    assert "tools/rollout/rollout.py" in inner_command
    assert str(config_path) in inner_command
    assert "sleep" not in joined


def test_dry_run_writes_configs_without_submitting(monkeypatch, tmp_path: Path) -> None:
    sweep = _load_sweep_module()
    base_config = _base_config(tmp_path)
    checkpoint = tmp_path / "checkpoint-200"
    checkpoint.mkdir()
    second_checkpoint = tmp_path / "checkpoint-400"
    second_checkpoint.mkdir()
    config_dir = tmp_path / "generated"
    output_root = tmp_path / "outputs"
    submitted = []

    monkeypatch.setattr(sweep, "SWEEP_CONFIG_DIR", config_dir)
    monkeypatch.setattr(sweep.subprocess, "run", lambda *args, **kwargs: submitted.append(args))

    result = sweep.main(
        [
            "--base-config",
            str(base_config),
            "--output-root",
            str(output_root),
            "--nproc",
            "2",
            "--ckpt",
            f"step-200={checkpoint}",
            "--ckpt",
            f"step-400={second_checkpoint}",
            "--dry-run",
        ]
    )

    generated = yaml.safe_load((config_dir / "step-200.yaml").read_text(encoding="utf-8"))
    second_generated = yaml.safe_load((config_dir / "step-400.yaml").read_text(encoding="utf-8"))
    assert result == 0
    assert submitted == []
    assert generated["model"]["checkpoint"] == str(checkpoint)
    assert generated["launcher"]["gpus"] == [0, 1]
    assert generated["output_dir"] == str(output_root / "step-200")
    assert second_generated["model"]["checkpoint"] == str(second_checkpoint)
    assert second_generated["launcher"]["gpus"] == [0, 1]
    assert second_generated["output_dir"] == str(output_root / "step-400")


def test_submit_command_limits_acp_job_name_length(tmp_path: Path) -> None:
    sweep = _load_sweep_module()

    command = sweep.build_submit_command(
        save_tag="checkpoint-" + "1" * 100,
        config_path=tmp_path / "config.yaml",
        nproc=4,
    )

    job_name = next(
        item.removeprefix("--job-name=") for item in command if item.startswith("--job-name=")
    )
    assert len(job_name) <= sweep.JOB_NAME_MAX_LEN


def test_output_root_can_be_overridden_by_environment(monkeypatch, tmp_path: Path) -> None:
    sweep = _load_sweep_module()
    monkeypatch.setenv("FLOW_FACTORY_ROLLOUT_OUTPUT_ROOT", str(tmp_path / "env-output"))

    assert sweep.resolve_output_root(None) == (tmp_path / "env-output").resolve()
    assert sweep.resolve_output_root(tmp_path / "cli-output") == (tmp_path / "cli-output").resolve()


def test_default_output_root_is_not_tied_to_original_owner_path() -> None:
    sweep = _load_sweep_module()

    assert sweep.resolve_output_root(None) == (Path.home() / "flow_factory_rollout").resolve()
    assert "/mnt/aigc/tuyouyuan/results/flow_factory_rollout" not in SWEEP_PATH.read_text(
        encoding="utf-8"
    )


def test_shell_wrapper_delegates_arguments_without_sleep() -> None:
    wrapper = REPO_ROOT / "tools" / "rollout" / "sweep_zimage_checkpoints.sh"
    content = wrapper.read_text(encoding="utf-8")

    assert "sweep_zimage_checkpoints.py" in content
    assert "BASE_CONFIG=" in content
    assert "CHECKPOINTS=(" in content
    assert "FLOW_FACTORY_AEC2_NAME" in content
    assert "FORWARD_ARGS=()" in content
    assert '"${PYTHON_ARGS[@]}"' in content
    assert "sleep" not in content


def test_default_checkpoint_list_contains_three_compatible_entries() -> None:
    sweep = _load_sweep_module()

    assert len(sweep.CHECKPOINTS) == 3
    assert [item["save_tag"] for item in sweep.CHECKPOINTS] == [
        "checkpoint-180",
        "human-hpsv3-checkpoint-280",
        "human-lighting-checkpoint-280",
    ]


def test_submit_command_honors_runtime_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    sweep = _load_sweep_module()
    monkeypatch.setenv("FLOW_FACTORY_WORKSPACE_NAME", "other-workspace")
    monkeypatch.setenv("FLOW_FACTORY_AEC2_NAME", "other-cluster")
    monkeypatch.setenv("FLOW_FACTORY_WS_SPEC", "N6lS.Iq.I10")

    command = sweep.build_submit_command(
        save_tag="step-180",
        config_path=tmp_path / "step-180.yaml",
        nproc=2,
    )

    assert "--workspace-name=other-workspace" in command
    assert "--aec2-name=other-cluster" in command
    assert "--worker-spec=N6lS.Iq.I10.2" in command
