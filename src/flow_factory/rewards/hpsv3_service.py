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

"""Pointwise reward client for the single-image HPSv3 HTTP service."""

from __future__ import annotations

import base64
import io
import math
from typing import Any, List, Optional, Tuple

import requests
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__)


class HPSv3ServiceRewardModel(PointwiseRewardModel):
    """Score prompt-image pairs through an HPSv3 ``POST /score`` service."""

    required_fields: Tuple[str, ...] = ("prompt", "image")
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator) -> None:
        super().__init__(config, accelerator)

        server_url = getattr(config, "server_url", None)
        if not server_url:
            raise ValueError("server_url is required for HPSv3ServiceRewardModel")

        self.server_url = str(server_url).rstrip("/")
        self.timeout = float(getattr(config, "timeout", 120.0))
        self.health_timeout = float(getattr(config, "health_timeout", 5.0))
        self.retry_attempts = int(getattr(config, "retry_attempts", 3))
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be at least 1")

        self._session = requests.Session()
        self._check_health()

    def _check_health(self) -> None:
        errors = []
        for path in ("/healthz", "/health"):
            try:
                response = self._session.get(
                    f"{self.server_url}{path}", timeout=self.health_timeout
                )
                response.raise_for_status()
                logger.info("Connected to HPSv3 reward server at %s", self.server_url)
                return
            except requests.RequestException as error:
                errors.append(f"{path}: {error}")

        raise RuntimeError(
            f"Cannot connect to HPSv3 reward server at {self.server_url}. " + "; ".join(errors)
        )

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        if not isinstance(image, Image.Image):
            raise TypeError(
                "HPSv3ServiceRewardModel expects PIL images because " "use_tensor_inputs is False"
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _score_one(self, prompt: str, image: Image.Image) -> float:
        payload = {
            "prompt": prompt,
            "image_base64": self._encode_image(image),
        }
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self._session.post(
                    f"{self.server_url}/score", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                if "score" not in data:
                    raise ValueError("response JSON does not contain 'score'")
                score = float(data["score"])
                if not math.isfinite(score):
                    raise ValueError(f"response score is not finite: {score}")
                return score
            except (requests.RequestException, TypeError, ValueError) as error:
                last_error = error
                if attempt < self.retry_attempts:
                    logger.warning(
                        "HPSv3 score request %d/%d failed: %s",
                        attempt,
                        self.retry_attempts,
                        error,
                    )

        raise RuntimeError(
            f"HPSv3 score request failed after {self.retry_attempts} attempts: " f"{last_error}"
        ) from last_error

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        **kwargs: Any,
    ) -> RewardModelOutput:
        """Score a batch of prompt-image pairs.

        Args:
            prompt: Text prompts aligned with ``image``.
            image: Generated PIL images to score.
            **kwargs: Unused sample fields supplied by the reward processor.

        Returns:
            Reward output containing one float32 scalar per input pair.
        """
        if image is None:
            raise ValueError("image is required for HPSv3ServiceRewardModel")
        if len(prompt) != len(image):
            raise ValueError(
                f"prompt and image batch lengths differ: {len(prompt)} != {len(image)}"
            )

        rewards = [
            self._score_one(sample_prompt, sample_image)
            for sample_prompt, sample_image in zip(prompt, image)
        ]
        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device)
        )
