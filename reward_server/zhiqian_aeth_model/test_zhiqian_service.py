#!/usr/bin/env python3
"""Smoke-test the HPSv3 HTTP reward service with one image."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_URL = "http://10.119.26.83:9010"
DEFAULT_IMAGE = (
    "/mnt/aigc/wangyubo/data/IG/eval_results/july_test/0723_test/u15/2k/U15_offical_mt50k/eval_full_20260731/images/qwenimagebench/0001_0.png"
)
DEFAULT_PROMPT = "test"


def normalize_base_url(url: str) -> str:
    return url.rstrip("/") + "/"


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return args.prompt


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, Any, float]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            elapsed = time.perf_counter() - started
            try:
                data = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                data = raw
            return resp.status, data, elapsed
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        try:
            data = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            data = raw
        return exc.code, data, elapsed


def check_health(base_url: str, timeout: float) -> bool:
    for endpoint in ("healthz", "health"):
        url = urljoin(base_url, endpoint)
        try:
            status, data, elapsed = request_json("GET", url, timeout=timeout)
        except URLError as exc:
            print(f"[health] GET /{endpoint} failed: {exc}", file=sys.stderr)
            continue

        print(f"[health] GET /{endpoint} -> HTTP {status} ({elapsed:.2f}s)")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if 200 <= status < 300:
            return True

    return False


def encode_image(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def score_image(base_url: str, image_path: Path, prompt: str, timeout: float) -> int:
    payload = {
        "prompt": prompt,
        "image_base64": encode_image(image_path),
    }
    url = urljoin(base_url, "score")
    status, data, elapsed = request_json("POST", url, payload=payload, timeout=timeout)

    print(f"[score] POST /score -> HTTP {status} ({elapsed:.2f}s)")
    print(f"[score] image: {image_path}")
    print(f"[score] prompt chars: {len(prompt)}")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    if not (200 <= status < 300):
        return 1
    if not isinstance(data, dict) or "score" not in data:
        print("[score] response does not contain a 'score' field", file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether an HPSv3 service is reachable and can score one image."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"default: {DEFAULT_URL}")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="image path to score")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"prompt paired with the image; default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="read prompt text from a UTF-8 file; overrides --prompt",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="skip /healthz and /health checks and call /score directly",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = normalize_base_url(args.url)
    image_path = Path(args.image)
    prompt = read_prompt(args)

    print(f"[config] url: {base_url.rstrip('/')}")
    print(f"[config] timeout: {args.timeout}s")

    try:
        if not args.skip_health:
            healthy = check_health(base_url, timeout=args.timeout)
            if not healthy:
                print("[health] no health endpoint returned HTTP 2xx; trying /score anyway")
        return score_image(base_url, image_path, prompt, timeout=args.timeout)
    except (FileNotFoundError, URLError, TimeoutError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# python server/zhiqian_aeth_model/test_hpsv3_service.py \
#     --url "http://10.119.23.161:9010" \
#     --image /mnt/aigc/wangyubo/code/IG/neo/flow_grpo_neo/work_dir/rollout/inference/reward_ab/hpsv3pp/samples/0021_id21/sample_00.png \
#     --prompt "整体为横版高密度乐高积木风乳制品商业信息图，约 4:3 比例。整个场景完全由可见凸点的塑料积木搭建，包括人物、牛奶包装、奶牛、生产线、建筑、文字牌、图标、边框和装饰植物。画面采用夸张超广角中心透视，中央牧场工作人员向镜头递出巨大牛奶盒，同时托举整箱产品；左右两侧由乳品生产线、货架和重复牛奶包装向远方延伸，形成沉浸式积木工厂。信息密度很高，但通过绿色、白色、黄色三个主要色系和模块化信息卡片保持清晰层级。\\n\\n## 整体构图\\n画面采用中心放射式结构。中央人物是主体，左手举起巨大竖版牛奶盒，右手托住横向整箱产品。两件商品明显向镜头突出，形成强烈近大远小关系。人物后方是纵深明显的积木乳品生产空间，左右货架和传送线向中央消失点收拢。\\n\\n外围使用绿色、白色积木搭建拱形框架，使场景像从一座积木乳品工厂入口向内部观看。下方增加牧场草地、奶牛、花朵和多个信息面板，使工业生产与牧场概念同时出现。\\n\\n## 左上品牌标识\\n左上角设置深绿色方形品牌牌，带米黄色细边框，内部使用白色中文。\\n\\n提取文字：\\n- 精选牧场\\n\\n## 顶部主标题\\n顶部中央使用超大型积木拼字标题。文字由白色和黄色积木组成，并安装在深绿色积木底板上。字体粗壮、立体、具有明显凸点和拼装缝隙，是画面最大的文字视觉中心。\\n\\n提取文字：\\n- 更快锁鲜\\n- 更好营养\\n\\n标题右上方增加绿色积木徽章与黄色文字，并配有星形装饰。\\n\\n提取文字：\\n- UP!\\n\\n主标题右侧加入向右上方弯曲的黄色积木箭头，强化提升、加速和积极向上的视觉方向。\\n\\n## 顶部右侧趣味标签\\n右上角放置绿色不规则圆角标签，配黄色粗体文字和一只黑白积木奶牛头像。整体类似趣味贴纸或儿童科普信息牌。\\n\\n提取文字：\\n- 好奶来自\\n- 好牧场！\\n\\n## 标题下方信息条\\n主标题下方横向放置黄色积木信息条，使用深绿色粗体文字。\\n\\n提取文字：\\n- 2小时新鲜直达 36城品质之选\\n\\n## 中央人物\\n中央是一名笑容明显的男性积木人偶，采用经典黄色塑料面部，黑色圆眼、黑色眉毛、微笑嘴形和少量胡茬点状印刷。人物佩戴浅卡其色积木帽，穿卡其色工作衬衫和深绿色背带工作服，腰部具有棕色腰带。\\n\\n人物姿态开放而热情，身体正对镜头。左手将巨大牛奶包装递向画面前方，右手托住整箱牛奶，使人物同时承担品牌工作人员、产品推荐者和视觉导览角色。\\n\\n## 左侧主商品\\n人物左手举起一盒尺寸极大的竖版积木牛奶包装。包装上半部分以白色积木为主，下半部分为深绿色，表面存在明显积木凸点和拼装结构。中央包含绿色品牌标识以及牧场、建筑和奶牛的线稿式图案。\\n\\n可辨识文字：\\n- 精选牧场\\n- “2小时”锁住牛奶新鲜\\n- 36城, 原生蛋白\\n- 纯牛奶\\n- 净含量:250mL\\n\\n商品因靠近镜头而成为画面最醒目的产品元素之一。\\n\\n## 右侧整箱商品\\n人物右手托住一只大型白色积木纸箱造型产品。箱体使用白色主体、绿色品牌块和牧场场景插画，三分之四角度朝向镜头。\\n\\n可辨识文字：\\n- 精选牧场\\n- 纯牛奶\\n- 36城 原生蛋白\\n\\n## 右侧营养信息板\\n右上至右中区域设置一块竖向白色信息卡，绿色积木边框，顶部为深绿色标题栏。内部划分为三个横向模块，每个模块左侧设置趣味图标，右侧为两级文字说明。\\n\\n提取文字：\\n- 纯牛奶 好营养\\n- 原生蛋白\\n- 优质乳蛋白\\n- 钙质丰富\\n- 助力骨骼强健\\n- 多重营养\\n- 均衡好吸收\\n\\n对应图标依次为弯曲手臂、骨骼和绿色盾牌，使营养卖点具有儿童科普卡片般的直观趣味性。\\n\\n## 左下新鲜信息板\\n左下区域设置高饱和黄色与绿色组合的趣味标题，旁边加入大型积木时钟图标。\\n\\n提取文字：\\n- 新鲜密码\\n- 2小时!\\n\\n其下方为白色竖向检查清单，每项左侧使用绿色勾选框。\\n\\n提取文字：\\n- 牧场直采\\n- 低温速达\\n- 锁住营养\\n- 口感更鲜\\n\\n## 牧场元素\\n左侧和底部布置多只黑白积木奶牛。奶牛采用明显方块化身体、积木腿和卡通头部。地面由绿色积木草坪构成，点缀白色、黄色和红色小花。牧场元素与后方工业生产线同时存在，形成从牧场到生产端的一体化视觉叙事。\\n\\n## 生产线与货架\\n背景左右两侧排列大量白绿色积木牛奶包装。左侧为多层货架，右侧为传送带和生产设备。包装尺寸随着距离逐渐缩小，并向中央消失点集中。工厂顶部由白色、灰色积木构成，加入黄色管线、绿色设备和多个品牌标识。\\n\\n## 底部中央信息条\\n底部中央设置横向绿色信息牌，左侧包含奶牛小图，右侧排列白色粗体文字，并加入一杯牛奶图标。\\n\\n提取文字：\\n- 每天一杯 活力满分\\n- 好营养 助成长\\n\\n下方连接蓝绿色横向品牌口号条。\\n\\n提取文字：\\n- 精选牧场 只为一杯好牛奶\\n\\n## 右下家庭场景\\n右下区域设置独立米白色积木信息卡，内部出现四名面带笑容的家庭成员积木人偶。人物以不同颜色服装和发型区分，形成欢乐家庭合影。\\n\\n提取文字：\\n- 全家都爱喝\\n- 更多安心选择\\n\\n## 右下产品保障图标条\\n家庭信息卡下方横向排列五个黄色底板的小型产品保障模块，每个模块上方有独立图标，下方配文字。图标依次表现水滴、瓶子、奶牛、雪花和放大镜。\\n\\n提取文字：\\n- 0添加香精\\n- 0添加防腐剂\\n- 生牛乳 ≥100%\\n- 全程冷链 保鲜\\n- 可追溯 源头\\n\\n## 色彩系统\\n画面以深绿色、草绿色、奶白色和亮黄色为核心。绿色代表牧场、品牌和天然感；白色对应牛奶、包装和工厂洁净环境；黄色承担标题重点、箭头、提示信息和经典积木人偶肤色。少量蓝色、棕色、红色和黑色用于人物服装、奶牛、花朵与底部信息条。\\n\\n## 材质与视觉语言\\n所有元素都应呈现真实积木塑料材质，具有清晰凸点、拼装缝隙、模块化边缘、圆润倒角和轻微镜面高光。大型文字同样由实体积木拼装而成，而不是普通二维印刷字。商品包装保持原有纸盒轮廓，但材质完全积木化。\\n\\n使用微缩模型摄影式照明，顶部和前方提供明亮柔和主光，塑料表面出现细腻高光。背景采用轻微景深，使中央人物、主牛奶盒、标题和营养信息板保持最高视觉清晰度。\\n\\n## 信息与视觉层级\\n第一层为“更快锁鲜 更好营养”大型积木标题；第二层为中央笑脸人物及巨大“纯牛奶”包装；第三层为“2小时新鲜直达 36城品质之选”、右侧营养卡和左下“新鲜密码 2小时!”；第四层为牧场奶牛、生产线、家庭人物以及产品保障图标。\\n\\n整体应呈现大型乳制品品牌与积木玩具世界结合的趣味商业广告效果，通过积木化人物、牛奶包装、奶牛、生产线、营养图标、时钟、箭头和家庭场景，把新鲜、营养、生产、运输和家庭消费信息转化成可以探索的微缩积木世界。"
