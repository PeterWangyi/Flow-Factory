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

"""Generate Qwen-Image images from a JSONL prompt file."""

import argparse

import torch
from _common import (
    add_common_arguments,
    attach_lora,
    load_prompt_records,
    validate_output_targets,
    write_batch_outputs,
)

from diffusers import QwenImagePipeline

DEFAULT_MODEL_PATH = "/mnt/aigc/zoemodels/Qwen-Image/Qwen-Image"


def parse_args() -> argparse.Namespace:
    """Parse Qwen-Image batch-inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        default_model_path=DEFAULT_MODEL_PATH,
        default_num_inference_steps=50,
    )
    parser.add_argument("--negative_prompt", "--negative-prompt", default=" ")
    return parser.parse_args()


def main() -> None:
    """Load Qwen-Image once and generate all selected prompts."""
    args = parse_args()
    records = load_prompt_records(
        args.prompt_file,
        prompt_key=args.prompt_key,
        raw_prompt=args.raw_prompt,
        start_index=args.start_index,
        max_samples=args.max_samples,
    )
    metadata_path = validate_output_targets(
        args.output_dir,
        records,
        overwrite=args.overwrite,
    )

    pipe = QwenImagePipeline.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    attach_lora(pipe, args.lora_path)
    if args.no_cpu_offload:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    def generate_image(prompt: str, seed: int):
        return pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    write_batch_outputs(
        records,
        args=args,
        model_name="QwenImagePipeline",
        generate_image=generate_image,
        metadata_path=metadata_path,
    )


if __name__ == "__main__":
    main()
