"""README 에 넣을 시연 이미지를 만든다.

- figure_pipeline.png : 원본 → 탐지 → 마스킹 결과를 가로로 붙인 것
- figure_compare.png  : 후보별 탐지 결과를 세로로 붙인 것 (놓친 얼굴이 보이도록)
- figure_sam.png      : 상자 블러 vs SAM 마스크 블러

사용:
  python make_figures.py --scene scene_00.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def jpg(name):
    return str(Path(name).with_suffix('.jpg'))


def label(img, text, h=28):
    """이미지 위에 제목 띠를 붙인다. 한글은 OpenCV 폰트가 못 그리므로 영문으로 쓴다."""
    bar = np.full((h, img.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, text, (8, h - 9), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def hstack(imgs, gap=8):
    h = max(i.shape[0] for i in imgs)
    pad = []
    for i in imgs:
        if i.shape[0] < h:
            i = np.vstack([i, np.full((h - i.shape[0], i.shape[1], 3), 245, np.uint8)])
        pad.append(i)
        pad.append(np.full((h, gap, 3), 245, np.uint8))
    return np.hstack(pad[:-1])


def vstack(imgs, gap=8):
    w = max(i.shape[1] for i in imgs)
    pad = []
    for i in imgs:
        if i.shape[1] < w:
            i = np.hstack([i, np.full((i.shape[0], w - i.shape[1], 3), 245, np.uint8)])
        pad.append(i)
        pad.append(np.full((gap, w, 3), 245, np.uint8))
    return np.vstack(pad[:-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--assets", default="assets")
    p.add_argument("--scene", default="scene_00.png")
    p.add_argument("--detector", default="yunet_face")
    p.add_argument("--candidates", default="baseline_haar,yunet_face,yolo_person")
    args = p.parse_args()

    data, res, assets = Path(args.data), Path(args.results), Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    s = args.scene

    # 1) 파이프라인 한 줄
    orig = cv2.imread(str(data / "images" / s))
    det = cv2.imread(str(res / args.detector / "vis" / jpg(s)))
    msk = cv2.imread(str(res / args.detector / "masked" / jpg(s)))
    if all(x is not None for x in (orig, det, msk)):
        fig = hstack([
            label(orig, "1. input (synthetic)"),
            label(det, f"2. detect - {args.detector}"),
            label(msk, "3. masked output"),
        ])
        cv2.imwrite(str(assets / "figure_pipeline.jpg"), fig)
        print("figure_pipeline.jpg")

    # 2) 후보 비교
    rows = []
    for c in args.candidates.split(","):
        v = cv2.imread(str(res / c.strip() / "vis" / jpg(s)))
        if v is not None:
            rows.append(label(v, f"{c.strip()}  (green=hit, red=MISS, blue=pred)"))
    if rows:
        cv2.imwrite(str(assets / "figure_compare.jpg"), vstack(rows))
        print("figure_compare.jpg")

    # 3) SAM 비교
    stem = Path(s).stem
    b = cv2.imread(str(res / "sam_refined" / f"{stem}_box.jpg"))
    sm = cv2.imread(str(res / "sam_refined" / f"{stem}_sam.jpg"))
    if b is not None and sm is not None:
        cv2.imwrite(str(assets / "figure_sam.jpg"),
                    hstack([label(b, "box blur"), label(sm, "SAM mask blur")]))
        print("figure_sam.jpg")


if __name__ == "__main__":
    main()
