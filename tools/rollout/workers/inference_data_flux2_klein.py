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

import torch
from _common import (
    add_common_arguments,
    attach_lora,
    load_prompt_records,
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


def main() -> None:
    """Load FLUX.2 Klein once and generate all selected prompts."""
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

    pipe = Flux2KleinPipeline.from_pretrained(
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
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    write_batch_outputs(
        records,
        args=args,
        model_name="Flux2KleinPipeline",
        generate_image=generate_image,
        metadata_path=metadata_path,
    )


if __name__ == "__main__":
    main()
