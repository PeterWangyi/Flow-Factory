"""HTTP service for Zhiqian's instruction-specialized HPSv3 models."""

from __future__ import annotations

import base64
import binascii
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import hpsv3_realism_inference as realism_inference
import zhiqian_model_sysprompt as system_prompts

from prompt_registry import resolve_instruction

DEFAULT_BASE_MODEL = realism_inference.DEFAULT_BASE_MODEL


APP_TITLE = "Zhiqian HPSv3 Reward Server"
APP_VERSION = "1.1.0"

_inferencer = None
_inferencer_lock = threading.Lock()
_score_lock = threading.Lock()


def _model_tag() -> str:
    tag = os.getenv("ZHIQIAN_MODEL_TAG", "").strip()
    if not tag:
        raise RuntimeError("ZHIQIAN_MODEL_TAG must be set")
    return tag


def _model_path() -> Path:
    value = os.getenv("ZHIQIAN_MODEL_PATH", "").strip()
    if not value:
        raise RuntimeError("ZHIQIAN_MODEL_PATH must be set")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Zhiqian model path does not exist: {path}")
    return path


def _image_base64_to_data_url(image_base64: str) -> str:
    value = (image_base64 or "").strip()
    if not value:
        raise ValueError("image_base64 must be non-empty")
    if value.startswith("data:image"):
        if "base64," not in value:
            raise ValueError("data:image URI must include a base64, payload")
        payload = value.split("base64,", 1)[1].strip()
    else:
        payload = value
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if value.startswith("data:image"):
        return value
    return f"data:image/png;base64,{payload}"


class SingleScoreRequest(BaseModel):
    class Config:
        extra = "forbid"

    prompt: str = ""
    image_base64: str = Field(..., min_length=1)

    @field_validator("image_base64")
    @classmethod
    def _validate_image_base64(cls, value: str) -> str:
        _image_base64_to_data_url(value)
        return value


class BatchScoreRequest(BaseModel):
    class Config:
        extra = "forbid"

    prompts: List[str] = Field(..., min_items=1)
    images_base64: List[str] = Field(..., min_items=1)

    @field_validator("images_base64")
    @classmethod
    def _validate_images_base64(cls, values: List[str]) -> List[str]:
        for value in values:
            _image_base64_to_data_url(value)
        return values


class SingleScoreResponse(BaseModel):
    score: float
    prompt: str


class BatchScoreResponse(BaseModel):
    scores: List[float]
    count: int


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Serve Zhiqian's instruction-specialized HPSv3 reward models.",
)


class ZhiqianRewardInferencer(realism_inference.HPSv3RewardInferencer):
    """Reuse the shared inference implementation with a selected instruction."""

    def __init__(self, *args, system_instruction: str, **kwargs):
        self.system_instruction = system_instruction
        super().__init__(*args, **kwargs)

    def _text(self, prompt: str) -> str:
        text = self.system_instruction
        prompt = str(prompt or "").strip()
        if prompt:
            text += "\n\nReference description or generation prompt:\n" + prompt
        return text + realism_inference.prompt_with_special_token


def _build_inferencer() -> ZhiqianRewardInferencer:
    model_tag = _model_tag()
    _, system_instruction = resolve_instruction(system_prompts, model_tag)
    model_path = _model_path()
    base_model = os.getenv("ZHIQIAN_BASE_MODEL", DEFAULT_BASE_MODEL)
    original_hpsv3 = os.getenv("ZHIQIAN_ORIGINAL_HPSV3") or None
    device = os.getenv("HPSV3_DEVICE", "cuda")
    model_format = os.getenv("ZHIQIAN_MODEL_FORMAT", "auto")
    batch_size = int(os.getenv("ZHIQIAN_BATCH_SIZE", "8"))
    return ZhiqianRewardInferencer(
        model_path=model_path,
        base_model=base_model,
        original_hpsv3=original_hpsv3,
        device=device,
        model_format=model_format,
        batch_size=batch_size,
        system_instruction=system_instruction,
    )


def get_inferencer() -> ZhiqianRewardInferencer:
    global _inferencer
    if _inferencer is None:
        with _inferencer_lock:
            if _inferencer is None:
                _inferencer = _build_inferencer()
    return _inferencer


@app.on_event("startup")
def warmup_inferencer() -> None:
    get_inferencer()


def _extract_score(value) -> float:
    if hasattr(value, "item"):
        if hasattr(value, "numel") and value.numel() > 1:
            return float(value.reshape(-1)[0].item())
        return float(value.item())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("Received an empty reward output")
        return _extract_score(value[0])
    return float(value)


def score_batch(prompts: List[str], images_base64: List[str]) -> List[float]:
    image_urls = [_image_base64_to_data_url(value) for value in images_base64]
    with _score_lock:
        rewards = get_inferencer().reward(
            prompts=prompts,
            image_paths=image_urls,
        )
    return [_extract_score(reward) for reward in rewards]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "model_tag": os.getenv("ZHIQIAN_MODEL_TAG", ""),
    }


@app.get("/healthz")
def healthz() -> dict[str, object]:
    model_tag = os.getenv("ZHIQIAN_MODEL_TAG", "")
    instruction_name, _ = resolve_instruction(system_prompts, model_tag)
    return {
        "status": "ok",
        "model_loaded": _inferencer is not None,
        "model_tag": model_tag,
        "model_path": os.getenv("ZHIQIAN_MODEL_PATH", ""),
        "inference": str(Path(realism_inference.__file__).resolve()),
        "instruction": instruction_name,
        "instruction_source": str(Path(system_prompts.__file__).resolve()),
    }


@app.post("/score", response_model=SingleScoreResponse)
def score(request: SingleScoreRequest) -> SingleScoreResponse:
    values = score_batch([request.prompt], [request.image_base64])
    return SingleScoreResponse(score=values[0], prompt=request.prompt)


@app.post("/scores", response_model=BatchScoreResponse)
def scores(request: BatchScoreRequest) -> BatchScoreResponse:
    if len(request.prompts) != len(request.images_base64):
        raise HTTPException(
            status_code=422,
            detail="`prompts` and `images_base64` must have the same length.",
        )
    values = score_batch(request.prompts, request.images_base64)
    return BatchScoreResponse(scores=values, count=len(values))
