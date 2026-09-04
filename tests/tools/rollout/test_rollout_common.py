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

"""Tests for shared rollout task and metadata behavior."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON_PATH = REPO_ROOT / "tools" / "rollout" / "workers" / "_common.py"


def _load_common_module():
    multiprocessing_module = types.ModuleType("torch.multiprocessing")
    multiprocessing_module.spawn = lambda *args, **kwargs: None
    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = "bfloat16"
    torch_module.multiprocessing = multiprocessing_module
    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = object
    module_names = ["torch", "torch.multiprocessing", "peft", "rollout_worker_common"]
    missing = object()
    original_modules = {name: sys.modules.get(name, missing) for name in module_names}
    try:
        sys.modules["torch"] = torch_module
        sys.modules["torch.multiprocessing"] = multiprocessing_module
        sys.modules["peft"] = peft_module
        spec = importlib.util.spec_from_file_location("rollout_worker_common", COMMON_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _records(common, count: int):
    return [
        common.PromptRecord(index=index, line_number=index + 1, prompt=f"prompt {index}")
        for index in range(count)
    ]


def test_one_prompt_four_images_uses_all_four_round_robin_shards() -> None:
    common = _load_common_module()
    tasks = common.build_image_tasks(
        _records(common, 1),
        num_images_per_prompt=4,
        base_seed=100,
    )

    shards = [common.shard_image_tasks(tasks, rank=rank, world_size=4) for rank in range(4)]

    assert [[task.sample_index for task in shard] for shard in shards] == [[0], [1], [2], [3]]
    assert [task.seed for task in tasks] == [100, 101, 102, 103]
    assert [task.image_name for task in tasks] == [
        "000000_000.png",
        "000000_001.png",
        "000000_002.png",
        "000000_003.png",
    ]


def test_four_prompts_one_image_use_all_four_round_robin_shards() -> None:
    common = _load_common_module()
    tasks = common.build_image_tasks(
        _records(common, 4),
        num_images_per_prompt=1,
        base_seed=42,
    )

    shards = [common.shard_image_tasks(tasks, rank=rank, world_size=4) for rank in range(4)]

    assert [[task.record.index for task in shard] for shard in shards] == [[0], [1], [2], [3]]
    assert [task.image_name for task in tasks] == [
        "000000.png",
        "000001.png",
        "000002.png",
        "000003.png",
    ]
    assert [task.seed for task in tasks] == [42, 42, 42, 42]


def test_metadata_shards_merge_in_prompt_sample_order(tmp_path: Path) -> None:
    common = _load_common_module()
    shard_0 = tmp_path / "metadata.rank-0.jsonl"
    shard_1 = tmp_path / "metadata.rank-1.jsonl"
    shard_0.write_text(
        json.dumps({"index": 1, "sample_index": 0, "image": "000001_000.png"}) + "\n",
        encoding="utf-8",
    )
    shard_1.write_text(
        json.dumps({"index": 0, "sample_index": 1, "image": "000000_001.png"})
        + "\n"
        + json.dumps({"index": 0, "sample_index": 0, "image": "000000_000.png"})
        + "\n",
        encoding="utf-8",
    )
    metadata_path = tmp_path / "metadata.jsonl"

    common.merge_metadata_shards([shard_0, shard_1], metadata_path)

    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    assert [(record["index"], record["sample_index"]) for record in records] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert not shard_0.exists()
    assert not shard_1.exists()


def test_cpu_offload_is_bound_to_worker_rank() -> None:
    common = _load_common_module()

    class FakePipeline:
        def __init__(self) -> None:
            self.offload_gpu_id = None

        def enable_model_cpu_offload(self, *, gpu_id: int) -> None:
            self.offload_gpu_id = gpu_id

    pipeline = FakePipeline()

    common.configure_pipeline_device(pipeline, cpu_offload=True, rank=3)

    assert pipeline.offload_gpu_id == 3


def test_write_batch_outputs_uses_task_names_and_records_physical_gpu(tmp_path: Path) -> None:
    common = _load_common_module()
    tasks = common.build_image_tasks(
        _records(common, 1),
        num_images_per_prompt=2,
        base_seed=20,
    )
    metadata_path = tmp_path / "metadata.rank-0.jsonl"
    args = Namespace(
        output_dir=tmp_path,
        model_path="/models/z-image",
        lora_path=None,
        height=1024,
        width=1024,
        num_inference_steps=40,
        guidance_scale=4.0,
    )

    class FakeImage:
        def save(self, path: Path) -> None:
            path.write_bytes(b"image")

    common.write_batch_outputs(
        tasks,
        args=args,
        model_name="ZImagePipeline",
        generate_image=lambda prompt, seed: FakeImage(),
        metadata_path=metadata_path,
        physical_gpu=5,
    )

    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    assert [record["image"] for record in metadata] == ["000000_000.png", "000000_001.png"]
    assert [record["seed"] for record in metadata] == [20, 21]
    assert [record["sample_index"] for record in metadata] == [0, 1]
    assert [record["gpu"] for record in metadata] == [5, 5]


def test_run_image_workers_propagates_spawn_failure(monkeypatch, tmp_path: Path) -> None:
    common = _load_common_module()
    tasks = common.build_image_tasks(
        _records(common, 2),
        num_images_per_prompt=1,
        base_seed=42,
    )

    def fail_spawn(*args, **kwargs):
        raise RuntimeError("rank failed")

    monkeypatch.setattr(common.mp, "spawn", fail_spawn)

    with pytest.raises(RuntimeError, match="rank failed"):
        common.run_image_workers(
            lambda rank, world_size, args, image_tasks: None,
            args=Namespace(num_gpus=2, gpu_ids="0,1", output_dir=tmp_path),
            tasks=tasks,
            metadata_path=tmp_path / "metadata.jsonl",
        )


def test_failed_overwrite_run_leaves_no_final_metadata(monkeypatch, tmp_path: Path) -> None:
    common = _load_common_module()
    tasks = common.build_image_tasks(
        _records(common, 1),
        num_images_per_prompt=1,
        base_seed=42,
    )
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text("old metadata\n", encoding="utf-8")

    def fail_spawn(*args, **kwargs):
        raise RuntimeError("rank failed")

    monkeypatch.setattr(common.mp, "spawn", fail_spawn)

    with pytest.raises(RuntimeError, match="rank failed"):
        common.run_image_workers(
            lambda rank, world_size, args, image_tasks: None,
            args=Namespace(num_gpus=2, gpu_ids="0,1", output_dir=tmp_path),
            tasks=tasks,
            metadata_path=metadata_path,
        )

    assert not metadata_path.exists()


def test_metadata_write_failure_leaves_no_partial_final_file(monkeypatch, tmp_path: Path) -> None:
    common = _load_common_module()
    tasks = common.build_image_tasks(
        _records(common, 1),
        num_images_per_prompt=1,
        base_seed=42,
    )
    metadata_path = tmp_path / "metadata.jsonl"

    def write_rank_metadata(rank, world_size, args, image_tasks):
        common.metadata_shard_path(args.output_dir, rank).write_text(
            '{"index": 0, "sample_index": 0}\n',
            encoding="utf-8",
        )

    def fail_json_dump(*args, **kwargs):
        raise OSError("metadata write failed")

    monkeypatch.setattr(common.json, "dumps", fail_json_dump)

    with pytest.raises(OSError, match="metadata write failed"):
        common.run_image_workers(
            write_rank_metadata,
            args=Namespace(num_gpus=1, gpu_ids="0", output_dir=tmp_path),
            tasks=tasks,
            metadata_path=metadata_path,
        )

    assert not metadata_path.exists()
    assert not (tmp_path / ".metadata.jsonl.tmp").exists()
    assert (tmp_path / "metadata.rank-0.jsonl").exists()
