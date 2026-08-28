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

from types import SimpleNamespace
from unittest.mock import patch

from flow_factory.hparams.args import _resolve_run_name
from flow_factory.trainers.loader import _synchronize_run_name


def test_run_name_timestamp_placeholder_is_resolved() -> None:
    resolved = _resolve_run_name(
        "qwen_image_hpsv3_grpo_2x8_{timestamp}",
        "unused_default",
        "20260826_123456",
    )

    assert resolved == "qwen_image_hpsv3_grpo_2x8_20260826_123456"


def test_explicit_run_name_without_placeholder_is_unchanged() -> None:
    resolved = _resolve_run_name(
        "stable_run_name",
        "unused_default",
        "20260826_123456",
    )

    assert resolved == "stable_run_name"


def test_missing_run_name_keeps_the_existing_generated_format() -> None:
    resolved = _resolve_run_name(
        None,
        "qwen-image_lora_grpo",
        "20260826_123456",
    )

    assert resolved == "qwen-image_lora_grpo_20260826_123456"


def test_run_name_is_synchronized_from_global_main_process() -> None:
    config = SimpleNamespace(
        log_args=SimpleNamespace(run_name="local_rank_name", save_dir="/tmp/flow-factory")
    )
    project_configuration = SimpleNamespace(project_dir=None, logging_dir=None)

    def set_directories(project_dir: str) -> None:
        project_configuration.project_dir = project_dir
        project_configuration.logging_dir = project_dir

    project_configuration.set_directories = set_directories
    accelerator = SimpleNamespace(
        is_main_process=False,
        project_configuration=project_configuration,
    )

    def broadcast(run_names: list[object], from_process: int) -> None:
        assert from_process == 0
        assert run_names == [None]
        run_names[0] = "rank_zero_run_20260826_123456"

    with patch(
        "flow_factory.trainers.loader.broadcast_object_list",
        side_effect=broadcast,
    ):
        _synchronize_run_name(config, accelerator)

    assert config.log_args.run_name == "rank_zero_run_20260826_123456"
    assert project_configuration.project_dir == (
        "/tmp/flow-factory/rank_zero_run_20260826_123456"
    )
    assert project_configuration.logging_dir == project_configuration.project_dir
