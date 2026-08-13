"""얼굴 마스킹 후보 탐지 — 후보 셋을 같은 자료로 비교한다.

후보는 크기가 아니라 종류로 벌렸다.
  baseline_haar : OpenCV Haar Cascade. 지금도 쓸 수 있는 고전 방식. 기준선.
  yunet_face    : 얼굴 전용으로 만들어진 모델 (OpenCV YuNet).
  yolo_person   : 범용 객체 탐지 모델의 person 클래스.

제외한 후보: 외부 클라우드 Vision API. 마스킹 전 원본을 외부로 보낼 수 없다.
성능을 재기 전에 조건으로 걸러냈다.

사용:
  python detect_faces.py --candidates baseline_haar,yunet_face,yolo_person
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------- 후보들

_cascade_cache = {}


def _load_cascade(name="haarcascade_frontalface_default.xml"):
    """cascade XML 을 메모리에서 읽어 로드한다.

    cv2.CascadeClassifier(경로) 는 Windows 에서 경로에 비ASCII 문자(예: 한글 사용자명)가
    있으면 조용히 빈 분류기를 돌려준다. 파일을 직접 읽어 FileStorage 로 넘기면 이를 피한다.
    """
    if name in _cascade_cache:
        return _cascade_cache[name]
    path = Path(cv2.data.haarcascades) / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} 없음. opencv-python 5.x 는 cascade XML 을 포함하지 않는다. "
            "`pip install 'opencv-python<5'` 로 맞춰라."
        )
    fs = cv2.FileStorage(path.read_text(encoding="utf-8"),
                         cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    c = cv2.CascadeClassifier()
    if not c.read(fs.getFirstTopLevelNode()) or c.empty():
        raise RuntimeError(f"cascade 로드 실패: {path}")
    _cascade_cache[name] = c
    return c


def detect_haar(img_bgr, cfg):
    """기준선. 신뢰도를 내주지 않으므로 이웃 수를 대용 점수로 쓴다."""
    cascade = _load_cascade()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    boxes, _, weights = cascade.detectMultiScale3(
        gray, scaleFactor=cfg["haar_scale"], minNeighbors=cfg["haar_neighbors"],
        minSize=(cfg["min_face_px"], cfg["min_face_px"]), outputRejectLevels=True,
    )
    out = []
    for (x, y, w, h), wt in zip(boxes, weights if len(weights) else [0] * len(boxes)):
        out.append({"box": [int(x), int(y), int(x + w), int(y + h)],
                    "score": float(wt), "src": "haar"})
    return out


_yunet = {}


def detect_yunet(img_bgr, cfg):
    """얼굴 전용 모델(YuNet). 짝마다 독립된 신뢰도를 주므로 임계값을 걸 수 있다."""
    h, w = img_bgr.shape[:2]
    key = (w, h, cfg["yunet_conf"], cfg["yunet_nms"])
    if key not in _yunet:
        _yunet.clear()  # 입력 크기가 바뀌면 새로 만든다
        _yunet[key] = cv2.FaceDetectorYN.create(
            cfg["yunet_weights"], "", (w, h),
            score_threshold=cfg["yunet_conf"], nms_threshold=cfg["yunet_nms"],
            top_k=cfg["yunet_topk"],
        )
    det = _yunet[key]
    det.setInputSize((w, h))
    _, faces = det.detect(img_bgr)
    out = []
    for f in faces if faces is not None else []:
        x, y, bw, bh = (float(v) for v in f[:4])
        out.append({
            "box": [max(0, int(x)), max(0, int(y)),
                    min(w, int(x + bw)), min(h, int(y + bh))],
            "score": float(f[14]), "src": "yunet",
        })
    return out


_yolo = {}


def detect_yolo(img_bgr, cfg):
    """범용 객체 탐지의 person 클래스. 얼굴이 아니라 사람 전체를 잡는다."""
    from ultralytics import YOLO

    if "m" not in _yolo:
        _yolo["m"] = YOLO(cfg["yolo_weights"])
    res = _yolo["m"].predict(img_bgr, conf=cfg["yolo_conf"], verbose=False)[0]
    out = []
    for b in res.boxes:
        if int(b.cls) != 0:  # 0 = person (COCO)
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        out.append({"box": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(b.conf), "src": "yolo_person"})
    return out


DETECTORS = {
    "baseline_haar": detect_haar,
    "yunet_face": detect_yunet,
    "yolo_person": detect_yolo,
}


# ---------------------------------------------------------------- 채점

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def covered(gt, pred):
    """정답 얼굴이 예측 상자에 '덮였는지'.

    이 업무의 판정은 상자가 정확히 맞느냐가 아니라 그 얼굴이 가려지느냐다.
    그래서 IoU 대신 정답 넓이 중 예측에 덮인 비율을 쓴다.
    사람 전체를 잡는 yolo_person 이 IoU 로는 부당하게 낮게 나오는 것을 피한다.
    """
    ix1, iy1 = max(gt[0], pred[0]), max(gt[1], pred[1])
    ix2, iy2 = min(gt[2], pred[2]), min(gt[3], pred[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return inter / area if area else 0.0


def score_image(gt_boxes, preds, cover_thr):
    """정답마다 덮였는지 보고, 어떤 정답도 덮지 않은 예측은 오탐으로 센다."""
    used = set()
    hits = []
    for g in gt_boxes:
        best_i, best_c = None, 0.0
        for i, p in enumerate(preds):
            c = covered(g, p["box"])
            if c > best_c:
                best_i, best_c = i, c
        if best_c >= cover_thr:
            hits.append({"gt": g, "cover": best_c, "pred_idx": best_i})
            used.add(best_i)
        else:
            hits.append({"gt": g, "cover": best_c, "pred_idx": None})
    misses = [h for h in hits if h["pred_idx"] is None]
    false_pos = [p for i, p in enumerate(preds) if i not in used]
    return hits, misses, false_pos


# ---------------------------------------------------------------- 시각화

def draw(img_bgr, gt_boxes, preds, hits, out_path):
    """눈으로 검산할 수 있는 형태로 남긴다.

    초록 = 잡힌 정답, 빨강 = 놓친 정답, 파랑 = 예측 상자.
    숫자만 보면 값이 의도한 것을 재고 있지 않은 경우를 놓친다.
    """
    vis = img_bgr.copy()
    for p in preds:
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 160, 0), 1)
        cv2.putText(vis, f"{p['score']:.2f}", (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 160, 0), 1)
    for h in hits:
        x1, y1, x2, y2 = h["gt"]
        ok = h["pred_idx"] is not None
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0) if ok else (0, 0, 255), 2)
        if not ok:
            cv2.putText(vis, "MISS", (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(str(out_path), vis)


def apply_mask(img_bgr, preds, out_path, mode="blur"):
    """후보 영역을 실제로 가린 결과물. 파이프라인의 최종 출력."""
    masked = img_bgr.copy()
    for p in preds:
        x1, y1, x2, y2 = p["box"]
        x1, y1 = max(0, x1), max(0, y1)
        roi = masked[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        if mode == "pixelate":
            small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_LINEAR)
            roi_out = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        else:
            k = max(3, (min(roi.shape[:2]) // 2) * 2 + 1)
            roi_out = cv2.GaussianBlur(roi, (k, k), 0)
        masked[y1:y2, x1:x2] = roi_out
    cv2.imwrite(str(out_path), masked)


# ---------------------------------------------------------------- 실행

def as_vis(path: Path, ext: str) -> Path:
    """시각화 산출물의 확장자. 검산용 그림이라 무손실일 필요가 없다."""
    return path.with_suffix(ext)


def nonclobber(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while (c := path.with_name(f"{path.stem}__{i}{path.suffix}")).exists():
        i += 1
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--candidates", default=",".join(DETECTORS))
    p.add_argument("--cover-thr", type=float, default=0.5,
                   help="정답 얼굴이 이 비율 이상 덮이면 잡은 것으로 센다")
    p.add_argument("--min-face-px", type=int, default=20)
    p.add_argument("--haar-scale", type=float, default=1.05)
    p.add_argument("--haar-neighbors", type=int, default=4)
    p.add_argument("--yunet-conf", type=float, default=0.3)
    p.add_argument("--yunet-nms", type=float, default=0.3)
    p.add_argument("--yunet-topk", type=int, default=5000)
    p.add_argument("--yunet-weights",
                   default="models/face_detection_yunet_2023mar.onnx")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--mask-mode", default="blur", choices=["blur", "pixelate"])
    p.add_argument("--vis-ext", default=".jpg", choices=[".jpg", ".png"],
                   help="시각화 산출물 형식. 저장소 용량 때문에 기본은 jpg")
    args = p.parse_args()

    cfg = vars(args)
    data = Path(args.data)
    gt_meta = json.loads((data / "ground_truth.json").read_text(encoding="utf-8"))
    outdir = Path(args.out)

    summary = {}
    for name in args.candidates.split(","):
        name = name.strip()
        fn = DETECTORS[name]
        vis_dir = outdir / name / "vis"
        mask_dir = outdir / name / "masked"
        vis_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        rows, tot_gt, tot_miss, tot_fp, tot_pred = [], 0, 0, 0, 0
        t0 = time.perf_counter()
        for item in gt_meta["images"]:
            img = cv2.imread(str(data / "images" / item["file"]))
            preds = fn(img, cfg)
            hits, misses, fps = score_image(item["gt_boxes"], preds, args.cover_thr)
            draw(img, item["gt_boxes"], preds, hits,
                 as_vis(vis_dir / item["file"], args.vis_ext))
            apply_mask(img, preds, as_vis(mask_dir / item["file"], args.vis_ext),
                       args.mask_mode)
            tot_gt += len(item["gt_boxes"]); tot_miss += len(misses)
            tot_fp += len(fps); tot_pred += len(preds)
            rows.append({
                "file": item["file"], "정답수": len(item["gt_boxes"]),
                "탐지수": len(preds), "놓침": len(misses), "오탐": len(fps),
                "놓친상자": [m["gt"] for m in misses],
                "놓친상자_최대덮임": [m["cover"] for m in misses],
                "예측": preds,
            })
        elapsed = time.perf_counter() - t0

        n_img = len(gt_meta["images"])
        m = {
            "총장수": n_img, "총정답얼굴": tot_gt, "총탐지": tot_pred,
            "놓침": tot_miss, "오탐": tot_fp,
            "재현율": (tot_gt - tot_miss) / tot_gt if tot_gt else None,
            "장당오탐": tot_fp / n_img if n_img else None,
            "총소요초": elapsed, "장당초": elapsed / n_img if n_img else None,
        }
        summary[name] = m
        print(f"\n=== {name} ===")
        print(f"  재현율 {m['재현율']:.4f}  ({tot_gt - tot_miss}/{tot_gt}), "
              f"놓침 {tot_miss}, 오탐 {tot_fp} (장당 {m['장당오탐']:.2f}), "
              f"장당 {m['장당초']:.3f}초")

        detail = nonclobber(outdir / f"{name}__detail.json")
        detail.write_text(
            json.dumps({"설정": {k: v for k, v in cfg.items()}, "집계": m, "건별": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    sp = nonclobber(outdir / "summary.json")
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n요약 → {sp}")


if __name__ == "__main__":
    main()
