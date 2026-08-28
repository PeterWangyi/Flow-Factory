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

import argparse

import torch
from diffusers import ZImagePipeline
from peft import PeftModel


DEFAULT_MODEL_PATH = "/mnt/aigc/zoemodels/Z-Image/Z-Image"
DEFAULT_CHECKPOINT = (
    "/mnt/aigc/wangyubo/code/IG/neo/RL/Flow-Factory/saves/z_image_hpsv3_2x8/"
    "[Z-Image]-data-u15human-reward-hpsv3-20260826_210707/checkpoints/checkpoint-180"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Z-Image with a PEFT LoRA.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--prompt", default="A cat holding a sign that says hello world")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="z_image_lora.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    """Generate one image with the Z-Image LoRA checkpoint."""
    args = _parse_args()

    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    if not args.no_lora:
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            args.checkpoint,
            torch_dtype=torch.bfloat16,
        )

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        cfg_normalization=False,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).images[0]
    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
