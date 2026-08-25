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
from unittest.mock import Mock

import pytest
import requests
import torch
from PIL import Image

from flow_factory.rewards.hpsv3_service import HPSv3ServiceRewardModel
from flow_factory.rewards.registry import get_reward_model_class


def _config(**overrides):
    values = {
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "server_url": "http://reward.example:9010/",
        "timeout": 12.0,
        "health_timeout": 2.0,
        "retry_attempts": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _accelerator():
    return SimpleNamespace(device=torch.device("cuda"))


def _response(payload=None):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def _reward(monkeypatch, scores=(0.25,)):
    session = Mock()
    session.get.return_value = _response({"status": "ok"})
    session.post.side_effect = [_response({"score": score}) for score in scores]
    monkeypatch.setattr(requests, "Session", lambda: session)
    return HPSv3ServiceRewardModel(_config(), _accelerator()), session


def test_registry_resolves_hpsv3_service_case_insensitively() -> None:
    assert get_reward_model_class("HPSV3_SERVICE") is HPSv3ServiceRewardModel


def test_scores_each_prompt_image_pair_with_service_contract(monkeypatch) -> None:
    reward, session = _reward(monkeypatch, scores=(0.25, 0.75))
    images = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]

    output = reward(prompt=["first", "second"], image=images)

    assert output.rewards.tolist() == pytest.approx([0.25, 0.75])
    assert output.rewards.device.type == "cpu"
    assert session.post.call_count == 2
    first_call = session.post.call_args_list[0]
    assert first_call.args == ("http://reward.example:9010/score",)
    assert first_call.kwargs["timeout"] == 12.0
    assert first_call.kwargs["json"]["prompt"] == "first"
    assert isinstance(first_call.kwargs["json"]["image_base64"], str)


def test_health_check_falls_back_to_health_endpoint(monkeypatch) -> None:
    session = Mock()
    unavailable = _response()
    unavailable.raise_for_status.side_effect = requests.HTTPError("not found")
    session.get.side_effect = [unavailable, _response({"status": "ok"})]
    monkeypatch.setattr(requests, "Session", lambda: session)

    HPSv3ServiceRewardModel(_config(), _accelerator())

    assert [call.args[0] for call in session.get.call_args_list] == [
        "http://reward.example:9010/healthz",
        "http://reward.example:9010/health",
    ]


def test_score_request_retries_transient_failure(monkeypatch) -> None:
    reward, session = _reward(monkeypatch)
    session.post.side_effect = [
        requests.ConnectionError("temporary"),
        _response({"score": 0.5}),
    ]

    output = reward(prompt=["prompt"], image=[Image.new("RGB", (4, 4))])

    assert output.rewards.item() == pytest.approx(0.5)
    assert session.post.call_count == 2


def test_rejects_non_finite_score(monkeypatch) -> None:
    reward, session = _reward(monkeypatch)
    session.post.side_effect = [_response({"score": "nan"})] * 3

    with pytest.raises(RuntimeError, match="not finite"):
        reward(prompt=["prompt"], image=[Image.new("RGB", (4, 4))])


def test_rejects_mismatched_batch_lengths(monkeypatch) -> None:
    reward, _ = _reward(monkeypatch)

    with pytest.raises(ValueError, match="batch lengths differ"):
        reward(prompt=["prompt"], image=[])
