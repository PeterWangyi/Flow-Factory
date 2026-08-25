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

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "prepare_aesthetics_prompts.py"
SPEC = importlib.util.spec_from_file_location("prepare_aesthetics_prompts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_dataset_extracts_nested_caption_and_preserves_plain_prompt(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "prepared"
    nested = json.dumps({"comprehensive_t2i_caption": "  concise caption  ", "unused": "large"})
    records = [
        {"prompt": nested, "id": 1},
        {"prompt": "plain prompt", "id": 2},
    ]
    _write_jsonl(input_dir / "train.jsonl", records)
    _write_jsonl(input_dir / "test.jsonl", records[:1])

    counts = MODULE.prepare_dataset(
        input_dir,
        output_dir,
        train_limit=None,
        test_limit=None,
        seed=42,
        overwrite=False,
    )

    assert counts == (2, 1)
    assert _read_jsonl(output_dir / "train.jsonl") == [
        {"prompt": "concise caption", "id": 1},
        {"prompt": "plain prompt", "id": 2},
    ]
    assert _read_jsonl(input_dir / "train.jsonl") == records


def test_limited_selection_is_deterministic_and_not_first_n(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    records = [{"prompt": f"prompt {index}", "id": index} for index in range(20)]
    _write_jsonl(input_dir / "train.jsonl", records)
    _write_jsonl(input_dir / "test.jsonl", records)

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    for output_dir in (first_output, second_output):
        MODULE.prepare_dataset(
            input_dir,
            output_dir,
            train_limit=5,
            test_limit=3,
            seed=7,
            overwrite=False,
        )

    first_train = _read_jsonl(first_output / "train.jsonl")
    assert first_train == _read_jsonl(second_output / "train.jsonl")
    assert len(first_train) == 5
    assert {record["id"] for record in first_train} != set(range(5))
    assert len(_read_jsonl(first_output / "test.jsonl")) == 3


def test_prepare_dataset_rejects_in_place_rewrite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        MODULE.prepare_dataset(
            tmp_path,
            tmp_path,
            train_limit=None,
            test_limit=None,
            seed=42,
            overwrite=False,
        )


def test_nested_prompt_requires_comprehensive_caption(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    _write_jsonl(source, [{"prompt": json.dumps({"other": "value"})}])

    with pytest.raises(ValueError, match="comprehensive_t2i_caption"):
        list(MODULE.iter_records(source))
