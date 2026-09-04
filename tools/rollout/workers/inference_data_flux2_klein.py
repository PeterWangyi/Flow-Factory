#!/usr/bin/env python3
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

"""Generate FLUX.2 Klein images from a JSONL prompt file."""

import argparse
from typing import List

import torch
from _common import (
    ImageTask,
    add_common_arguments,
    attach_lora,
    build_image_tasks,
    configure_pipeline_device,
    load_prompt_records,
    metadata_shard_path,
    resolve_physical_gpu_ids,
    run_image_workers,
    shard_image_tasks,
    validate_output_targets,
    write_batch_outputs,
)

from diffusers import Flux2KleinPipeline

DEFAULT_MODEL_PATH = (
    "/mnt/aigc/shared_env/huggingface/hub/"
    "models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/"
    "a3b4f4849157f664bdbc776fd7453c2783562f4d"
)


def parse_args() -> argparse.Namespace:
    """Parse FLUX.2 Klein batch-inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        default_model_path=DEFAULT_MODEL_PATH,
        default_num_inference_steps=28,
    )
    return parser.parse_args()


def _run_rank(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    tasks: List[ImageTask],
) -> None:
    """Load one FLUX.2 Klein replica and generate one interleaved task shard."""
    torch.cuda.set_device(rank)
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    attach_lora(pipe, args.lora_path)
    configure_pipeline_device(pipe, cpu_offload=not args.no_cpu_offload, rank=rank)
    rank_tasks = shard_image_tasks(tasks, rank=rank, world_size=world_size)
    physical_gpu = resolve_physical_gpu_ids(args)[rank]

    def generate_image(prompt: str, seed: int):
        return pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device=f"cuda:{rank}").manual_seed(seed),
        ).images[0]

    write_batch_outputs(
        rank_tasks,
        args=args,
        model_name="Flux2KleinPipeline",
        generate_image=generate_image,
        metadata_path=metadata_shard_path(args.output_dir, rank),
        physical_gpu=physical_gpu,
    )


def main() -> None:
    """Generate selected FLUX.2 Klein prompt/sample tasks across visible GPUs."""
    args = parse_args()
    records = load_prompt_records(
        args.prompt_file,
        prompt_key=args.prompt_key,
        raw_prompt=args.raw_prompt,
        start_index=args.start_index,
        max_samples=args.max_samples,
    )
    tasks = build_image_tasks(
        records,
        num_images_per_prompt=args.num_images_per_prompt,
        base_seed=args.seed,
    )
    metadata_path = validate_output_targets(
        args.output_dir,
        tasks,
        overwrite=args.overwrite,
    )
    run_image_workers(
        _run_rank,
        args=args,
        tasks=tasks,
        metadata_path=metadata_path,
    )


if __name__ == "__main__":
    main()
