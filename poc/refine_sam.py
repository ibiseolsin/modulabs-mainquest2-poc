"""SAM 세그멘테이션으로 상자를 정밀 마스크로 다듬는다.

탐지가 내주는 것은 직사각형이다. 직사각형을 그대로 블러하면 배경까지 뭉개져
사진의 쓸모가 줄어든다. SAM 에 상자를 프롬프트로 주면 그 안에서 사람 영역만
분리해 준다. 가리는 넓이가 줄고 사진이 살아난다.

이 단계는 성공 기준의 판정 항목이 아니다. 탐지가 기준을 넘은 뒤에 붙이는 단계다.

사용:
  python refine_sam.py --detector mediapipe_face --limit 4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_boxes(results_dir, detector):
    path = Path(results_dir) / f"{detector}__detail.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return {r["file"]: [p["box"] for p in r["예측"]] for r in d["건별"]}


def blur_with_mask(img_bgr, mask_bool, ksize_ratio=0.06):
    """마스크가 참인 화소만 블러로 갈아 끼운다."""
    h, w = img_bgr.shape[:2]
    k = max(3, int(min(h, w) * ksize_ratio) | 1)
    blurred = cv2.GaussianBlur(img_bgr, (k, k), 0)
    out = img_bgr.copy()
    out[mask_bool] = blurred[mask_bool]
    return out


def box_mask(shape, boxes):
    m = np.zeros(shape[:2], dtype=bool)
    for x1, y1, x2, y2 in boxes:
        m[max(0, y1):y2, max(0, x1):x2] = True
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--detector", default="mediapipe_face")
    p.add_argument("--sam-weights", default="mobile_sam.pt")
    p.add_argument("--out", default="results/sam_refined")
    p.add_argument("--limit", type=int, default=0, help="0 이면 전체")
    p.add_argument("--blur-ratio", type=float, default=0.06)
    args = p.parse_args()

    from ultralytics import SAM

    boxes_by_file = load_boxes(args.results, args.detector)
    sam = SAM(args.sam_weights)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(boxes_by_file)
    if args.limit:
        files = files[: args.limit]

    stats = []
    for fn in files:
        boxes = boxes_by_file[fn]
        img = cv2.imread(str(Path(args.data) / "images" / fn))
        if img is None or not boxes:
            continue

        res = sam.predict(img, bboxes=boxes, verbose=False)[0]
        if res.masks is None:
            print(f"  {fn}: 마스크 없음, 건너뜀")
            continue
        seg = np.zeros(img.shape[:2], dtype=bool)
        for m in res.masks.data.cpu().numpy():
            seg |= cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        rect = box_mask(img.shape, boxes)
        cv2.imwrite(str(outdir / f"{Path(fn).stem}_box.jpg"),
                    blur_with_mask(img, rect, args.blur_ratio))
        cv2.imwrite(str(outdir / f"{Path(fn).stem}_sam.jpg"),
                    blur_with_mask(img, seg, args.blur_ratio))

        total = img.shape[0] * img.shape[1]
        s = {"file": fn, "상자마스크_화소비": float(rect.sum() / total),
             "SAM마스크_화소비": float(seg.sum() / total),
             "줄어든비율": float(1 - seg.sum() / rect.sum()) if rect.sum() else None}
        stats.append(s)
        print(f"  {fn}: 상자 {s['상자마스크_화소비']:.4f} → SAM {s['SAM마스크_화소비']:.4f} "
              f"(가리는 넓이 {s['줄어든비율']*100:.1f}% 감소)")

    (outdir / "sam_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if stats:
        avg = sum(s["줄어든비율"] for s in stats) / len(stats)
        print(f"\n평균 가리는 넓이 감소: {avg*100:.1f}%")


if __name__ == "__main__":
    main()
