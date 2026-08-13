"""docs/PITFALLS.md 에 넣을 증거 이미지를 재현한다.

두 실수를 일부러 다시 만들어 남긴다. 고치고 나면 왜 그게 문제였는지가 안 보이기 때문이다.

  pitfall_bad_faces.png : 얼굴을 256px 로 직접 생성했을 때 (정답이 얼굴이 아니었던 상태)
  pitfall_gt_toobig.png : 정답을 붙여넣은 정사각형 전체로 잡았을 때 (멀쩡한 탐지가 놓침으로 세어진 상태)

사용:
  python make_pitfall_figures.py --out ../docs/assets
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import build_dataset as bd
import detect_faces as df


def fig_bad_faces(out_path, seed, device):
    """256px 로 얼굴을 생성해 합성한다. 당시의 버그를 그대로 재현한다."""
    import torch

    pipe = bd.build_pipe("stable-diffusion-v1-5/stable-diffusion-v1-5", device)

    faces = []
    for i in range(5):
        prompt = bd.FACE_PROMPT.format(v=bd.FACE_VARIANTS[i])
        g = torch.Generator(device=pipe.device).manual_seed(seed + i)
        # 여기가 버그였다. SD1.5 는 512 로 학습돼 있는데 256 으로 직접 생성했다.
        faces.append(pipe(prompt, negative_prompt=bd.NEG, num_inference_steps=25,
                          guidance_scale=7.5, height=256, width=256,
                          generator=g).images[0])

    g = torch.Generator(device=pipe.device).manual_seed(seed + 1000)
    scene = pipe(bd.SCENE_PROMPTS[0], negative_prompt=bd.NEG, num_inference_steps=25,
                 guidance_scale=7.5, height=512, width=768, generator=g).images[0]

    rng = random.Random(seed)
    img, gt, _ = bd.composite(scene, faces, rng, 5, 48, 150)
    vis = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for b in gt:
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 200, 0), 2)
    cv2.imwrite(str(out_path), vis)
    print(f"{out_path.name}: 정답 {len(gt)}개, 이 중 대부분이 얼굴이 아니다")


def fig_gt_toobig(out_path, data, results, scene):
    """정답을 붙여넣은 정사각형으로 되돌려 그린다. 예측은 지금 것을 그대로 쓴다."""
    meta = json.loads((Path(data) / "ground_truth.json").read_text(encoding="utf-8"))
    item = next(i for i in meta["images"] if i["file"] == scene)
    detail = json.loads(
        (Path(results) / "yunet_face__detail.json").read_text(encoding="utf-8"))
    preds = next(r for r in detail["건별"] if r["file"] == scene)["예측"]

    # 당시의 잘못된 정답: 얼굴 영역이 아니라 붙여넣은 정사각형 전체
    wrong_gt = [tuple(b) for b in item["paste_boxes"]]
    hits, misses, _ = df.score_image(wrong_gt, preds, 0.5)

    img = cv2.imread(str(Path(data) / "images" / scene))
    df.draw(img, wrong_gt, preds, hits, out_path)
    print(f"{out_path.name}: 정답 {len(wrong_gt)}개 중 {len(misses)}개가 "
          f"놓침으로 세어짐 (예측은 얼굴에 붙어 있는데도)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="../docs/assets")
    p.add_argument("--scene", default="scene_03.png")
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--device", default=None)
    p.add_argument("--skip-diffusion", action="store_true",
                   help="SD 재생성 없이 두 번째 그림만 만든다")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_gt_toobig(out / "pitfall_gt_toobig.jpg", args.data, args.results, args.scene)

    if not args.skip_diffusion:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        fig_bad_faces(out / "pitfall_bad_faces.jpg", args.seed, device)


if __name__ == "__main__":
    main()
