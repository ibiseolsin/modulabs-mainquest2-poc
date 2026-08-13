"""합성 테스트셋 생성 — Stable Diffusion 으로 강의장 배경과 얼굴을 만들고,
얼굴을 알려진 좌표에 합성해 정답 상자를 기록한다.

실제 수강생 사진을 쓰지 않고 검증하기 위한 구성이다. 붙인 좌표를 알고 있으므로
장면마다 정답을 다시 찍을 필요가 없고, 모델을 돌리기 전에 정답이 확정된다.

정답은 붙여넣은 정사각형이 아니라 그 안의 얼굴 영역(FACE_BOXES)이다.
그 비율은 얼굴 풀 8장을 눈으로 보고 손으로 적었다. 자세한 것은 docs/PITFALLS.md 2번.

검산용으로 세 가지를 함께 남긴다.
  data/face_pool/, face_pool_contact.png : 붙이기 전 얼굴이 정말 얼굴인지
  data/gt_vis/                           : 정답 상자가 얼굴에 맞는지

사용:
  python build_dataset.py --n-scenes 12 --out data
"""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageFilter

SCENE_PROMPTS = [
    "wide photo of an empty modern classroom with rows of desks and a projector screen, daylight",
    "photo of an empty coworking space with long tables, laptops closed, large windows",
    "photo of an empty seminar hall with chairs facing a stage, warm lighting",
    "photo of an empty office meeting room with a whiteboard covered in diagrams",
    "photo of an empty university lecture hall, tiered seating, morning light",
    "photo of an empty startup office lounge with sofas and a bookshelf",
]

# 얼굴은 512px 로 만든 뒤 붙일 크기로 줄인다.
# SD1.5 는 512 로 학습돼 있어 256 으로 직접 생성하면 얼굴이 아니라 노이즈가 나온다.
# (이 문제를 눈으로 확인하지 않았으면 얼굴이 아닌 것을 정답으로 놓고 채점할 뻔했다.)
FACE_GEN_PX = 512

# 연령·성별·안경·수염을 흔들어 둔다. 비슷한 얼굴만 모으면 후보들이 전부 비슷해 보인다.
FACE_VARIANTS = [
    "a young man with short black hair",
    "an older man with grey hair and glasses",
    "a young woman with long dark hair",
    "a middle aged woman with short hair",
    "a man with a beard wearing a cap",
    "a woman wearing glasses",
    "a teenage boy",
    "an elderly woman smiling",
]

FACE_PROMPT = (
    "extreme closeup face portrait photo of {v}, face fills the entire frame, "
    "cropped at chin and forehead, front facing, sharp focus, natural lighting"
)

# 생성된 얼굴 안에서 '얼굴'이 실제로 차지하는 영역. 0~1 로 정규화한 (x1,y1,x2,y2).
# data/face_pool/ 의 8장을 격자와 함께 띄워 놓고 눈으로 읽어 적었다. FACE_VARIANTS 와 같은 순서다.
#
# 왜 필요한가: 붙여넣는 정사각형에는 머리카락과 인물 사진의 배경이 함께 들어 있다.
# 정사각형 전체를 정답으로 놓으면, 얼굴을 정확히 찾은 탐지기도 '덮임 비율'이 0.5 를 못 넘어
# 놓친 것으로 세어진다. 실제로 처음에 이렇게 채점해서 YuNet 이 50% 로 나왔다.
# 정답은 모델이 아니라 사람이 정한다. 그래서 손으로 적었다.
FACE_BOXES = [
    (0.00, 0.00, 0.55, 0.95),  # 0 젊은 남자 (측면으로 잘린 구도)
    (0.22, 0.00, 0.90, 0.92),  # 1 안경 쓴 노년 남자
    (0.28, 0.00, 0.75, 0.88),  # 2 긴 머리 젊은 여자
    (0.42, 0.05, 0.85, 0.87),  # 3 단발 중년 여자
    (0.00, 0.00, 0.68, 1.00),  # 4 모자 쓴 수염 남자
    (0.22, 0.00, 0.85, 0.97),  # 5 안경 쓴 여자
    (0.18, 0.00, 0.90, 1.00),  # 6 십대 남자
    (0.30, 0.05, 0.85, 0.95),  # 7 웃는 노년 여자
]

NEG = ("cartoon, illustration, painting, watermark, text, blurry, deformed, "
       "extra limbs, multiple people, body, torso")


def build_pipe(model_id, device):
    import torch
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def gen_faces(pipe, n, seed):
    """서로 다른 얼굴 n개를 만든다. seed 를 고정해 재현 가능하게 한다."""
    import torch

    faces = []
    for i in range(n):
        prompt = FACE_PROMPT.format(v=FACE_VARIANTS[i % len(FACE_VARIANTS)])
        g = torch.Generator(device=pipe.device).manual_seed(seed + i)
        img = pipe(
            prompt, negative_prompt=NEG, num_inference_steps=25, guidance_scale=7.5,
            height=FACE_GEN_PX, width=FACE_GEN_PX, generator=g,
        ).images[0]
        faces.append(img)
    return faces


def gen_scenes(pipe, n, seed, w=768, h=512):
    import torch

    scenes = []
    for i in range(n):
        g = torch.Generator(device=pipe.device).manual_seed(seed + 1000 + i)
        img = pipe(
            SCENE_PROMPTS[i % len(SCENE_PROMPTS)], negative_prompt=NEG,
            num_inference_steps=25, guidance_scale=7.5, height=h, width=w, generator=g,
        ).images[0]
        scenes.append(img)
    return scenes


def oval_mask(size):
    """붙여넣기 경계를 부드럽게 하기 위한 타원 마스크."""
    from PIL import ImageDraw

    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).ellipse([0, 0, size[0] - 1, size[1] - 1], fill=255)
    return m.filter(ImageFilter.GaussianBlur(size[0] * 0.04))


def overlaps(box, boxes, pad=8):
    x1, y1, x2, y2 = box
    for b in boxes:
        if not (x2 + pad < b[0] or b[2] + pad < x1 or y2 + pad < b[1] or b[3] + pad < y1):
            return True
    return False


def composite(scene, faces, rng, n_faces, min_px, max_px):
    """얼굴을 겹치지 않게 붙이고, 붙인 좌표를 정답으로 반환한다.

    크기를 넓게 흔들어 '뒷줄의 작은 얼굴'을 일부러 만든다.
    쉬운 것만 모으면 후보들이 전부 비슷해 보여 표가 아무것도 알려 주지 않는다.
    """
    canvas = scene.convert("RGB").copy()
    W, H = canvas.size
    pasted, gt = [], []
    # 한 장면 안에서는 서로 다른 얼굴을 쓴다. 같은 얼굴이 반복되면
    # 그 얼굴 하나에 대한 판정을 여러 번 세는 셈이라 표가 부풀려진다.
    order = rng.sample(range(len(faces)), k=min(n_faces, len(faces)))
    for slot in range(n_faces):
        for _attempt in range(60):
            s = rng.randint(min_px, max_px)
            x = rng.randint(0, max(0, W - s))
            y = rng.randint(0, max(0, H - s))
            box = (x, y, x + s, y + s)
            if not overlaps(box, [b for b, _ in pasted]):
                break
        else:
            continue  # 자리를 못 찾으면 이 얼굴은 건너뛴다
        idx = order[slot % len(order)]
        f = faces[idx].resize((s, s), Image.LANCZOS)
        # 작은 얼굴은 원본 사진에서도 흐릿하다. 크기에 따라 약한 블러를 준다.
        if s < 64:
            f = f.filter(ImageFilter.GaussianBlur(0.6))
        canvas.paste(f, (x, y), oval_mask((s, s)))
        pasted.append((box, idx))

        # 정답은 붙여넣은 정사각형이 아니라 그 안의 얼굴 영역이다.
        fx1, fy1, fx2, fy2 = FACE_BOXES[idx % len(FACE_BOXES)]
        gt.append((round(x + fx1 * s), round(y + fy1 * s),
                   round(x + fx2 * s), round(y + fy2 * s)))
    return canvas, gt, pasted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-scenes", type=int, default=12)
    p.add_argument("--n-face-pool", type=int, default=8)
    p.add_argument("--faces-per-scene", type=int, default=5)
    p.add_argument("--min-face-px", type=int, default=48,
                   help="붙여넣는 정사각형의 최소 변. 실제 얼굴은 이보다 작다(FACE_BOXES 참고)")
    p.add_argument("--max-face-px", type=int, default=150)
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="data")
    args = p.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={args.model}")

    outdir = Path(args.out)
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    pipe = build_pipe(args.model, device)
    print(f"얼굴 {args.n_face_pool}개 생성 중...")
    faces = gen_faces(pipe, args.n_face_pool, args.seed)

    # 얼굴 풀을 그대로 남긴다. 붙이기 전에 이것이 정말 얼굴인지 눈으로 확인해야 한다.
    pool_dir = outdir / "face_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    contact = Image.new("RGB", (256 * len(faces), 256), "white")
    for i, f in enumerate(faces):
        f.save(pool_dir / f"face_{i:02d}.png")
        contact.paste(f.resize((256, 256), Image.LANCZOS), (256 * i, 0))
    contact.save(outdir / "face_pool_contact.png")
    print(f"배경 {args.n_scenes}장 생성 중...")
    scenes = gen_scenes(pipe, args.n_scenes, args.seed)

    rng = random.Random(args.seed)
    manifest = []
    for i, scene in enumerate(scenes):
        n = args.faces_per_scene
        img, gt, pasted = composite(scene, faces, rng, n,
                                    args.min_face_px, args.max_face_px)
        name = f"scene_{i:02d}.png"
        img.save(outdir / "images" / name)

        # 정답을 그려 둔다. 채점표가 틀리면 그 표로는 아무것도 판단할 수 없으므로,
        # 모델을 돌리기 전에 정답부터 눈으로 확인해야 한다.
        from PIL import ImageDraw

        gt_vis = img.copy()
        dr = ImageDraw.Draw(gt_vis)
        for b in gt:
            dr.rectangle(b, outline=(0, 200, 0), width=2)
        gt_dir = outdir / "gt_vis"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_vis.save((gt_dir / name).with_suffix(".jpg"), quality=90)

        manifest.append({
            "file": name, "gt_boxes": gt, "n_faces": len(gt),
            # 붙여넣은 정사각형과 얼굴 풀 번호. 정답 정의를 나중에 되짚을 때 쓴다.
            "paste_boxes": [list(b) for b, _ in pasted],
            "pool_idx": [i for _, i in pasted],
        })
        print(f"  {name}: 얼굴 {len(gt)}개")

    meta = {
        "생성설정": vars(args) | {"device": device},
        "총장수": len(manifest),
        "총얼굴수": sum(m["n_faces"] for m in manifest),
        "images": manifest,
    }
    gt_path = outdir / "ground_truth.json"
    gt_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n정답 → {gt_path}  (총 {meta['총얼굴수']}개 얼굴 / {meta['총장수']}장)")


if __name__ == "__main__":
    main()
