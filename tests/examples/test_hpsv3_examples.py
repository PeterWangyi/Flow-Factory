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

from pathlib import Path

import pytest

from flow_factory.hparams import Arguments
from flow_factory.rewards.hpsv3_service import HPSv3ServiceRewardModel
from flow_factory.rewards.registry import get_reward_model_class

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("path", "model_type", "config_file", "train_steps", "eval_steps"),
    (
        (
            "examples/grpo/lora/flux2/hpsv3_2x8.yaml",
            "flux2",
            "config/accelerate_configs/fsdp2.yaml",
            8,
            28,
        ),
        (
            "examples/grpo/lora/qwen_image/hpsv3_2x8.yaml",
            "qwen-image",
            "config/accelerate_configs/fsdp2.yaml",
            10,
            50,
        ),
        (
            "examples/grpo/lora/z_image/hpsv3_2x8.yaml",
            "z-image",
            "config/deepspeed/deepspeed_zero2.yaml",
            10,
            40,
        ),
        (
            "examples/grpo/lora/sd3_5/hpsv3_2x8_medium.yaml",
            "sd3-5",
            "config/deepspeed/deepspeed_zero2.yaml",
            10,
            40,
        ),
    ),
)
def test_hpsv3_two_node_examples_parse(
    path: str,
    model_type: str,
    config_file: str,
    train_steps: int,
    eval_steps: int,
) -> None:
    config = Arguments.load_from_yaml(str(ROOT / path))
    reward = config.reward_args[0]

    assert config.model_args.model_type == model_type
    assert config.config_file == config_file
    assert config.num_processes == 16
    assert config.num_machines == 2
    assert config.data_args.preprocess_parallelism == "local"
    assert config.data_args.sampler_type == "group_contiguous"
    assert config.training_args.num_inference_steps == train_steps
    assert config.eval_args.num_inference_steps == eval_steps
    assert config.training_args.group_size == 16
    assert config.training_args.unique_sample_num_per_epoch == 48
    assert reward.reward_model == "hpsv3_service"
    assert reward.batch_size == 1
    assert reward.async_reward is True
    assert reward.num_workers == 1
    assert reward.applicable_datasets == ["aesthetics"]


def test_hpsv3_service_is_registered() -> None:
    assert get_reward_model_class("HPSV3_SERVICE") is HPSv3ServiceRewardModel
