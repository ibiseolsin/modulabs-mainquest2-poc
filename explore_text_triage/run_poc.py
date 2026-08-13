"""문의 분류 PoC — 후보 3개를 같은 자료로 비교한다.

사용:
  python run_poc.py --data data/tickets.csv --out results
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

LABELS = ["결제", "일정", "과제", "기술", "기타"]

KEYWORD_RULES = {
    "결제": ["결제", "환불", "수강료", "영수증", "청구", "카드"],
    "일정": ["언제", "개강", "마감", "연장", "예약", "발표", "일정"],
    "과제": ["과제", "채점", "제출", "피드백", "재제출"],
    "기술": ["에러", "오류", "안 되", "안되", "로그인", "재생", "다운", "링크", "로딩"],
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def predict_baseline(rows, _):
    """기준선: 지금 담당자가 쓰는 키워드 규칙."""
    out = []
    for r in rows:
        hits = {
            lab: sum(1 for k in kws if k in r["text"]) for lab, kws in KEYWORD_RULES.items()
        }
        best = max(hits, key=hits.get)
        top = hits[best]
        total = sum(hits.values())
        label = best if top > 0 else "기타"
        score = top / total if total else 0.0
        out.append((label, score, hits))
    return out


def predict_tfidf(rows, cfg):
    """후보 A: 사내 자료로 학습한 전용 분류기 (문자 n-gram TF-IDF + 로지스틱)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    texts = [r["text"] for r in rows]
    y = [r["gold_label"] for r in rows]
    out = [None] * len(rows)
    # 자료가 20건뿐이므로 leave-one-out 으로 매 건을 학습에서 빼고 예측한다.
    for train_idx, (test_idx,) in (
        (tr, te) for tr, te in LeaveOneOut().split(texts)
    ):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        X = vec.fit_transform([texts[i] for i in train_idx])
        clf = LogisticRegression(max_iter=2000, C=cfg["tfidf_C"])
        clf.fit(X, [y[i] for i in train_idx])
        proba = clf.predict_proba(vec.transform([texts[test_idx]]))[0]
        j = int(proba.argmax())
        out[test_idx] = (clf.classes_[j], float(proba[j]), None)
    return out


def predict_embedding(rows, cfg):
    """후보 B: 다국어 임베딩 + 라벨 설명문과의 코사인 유사도.

    학습 자료가 없어도 되는 구성. 점수는 짝마다 독립된 코사인 값이라 임계값을 걸 수 있다.
    """
    from sentence_transformers import SentenceTransformer

    label_desc = {
        "결제": "수강료 결제, 환불, 영수증, 카드 청구에 관한 문의",
        "일정": "개강일, 마감일, 예약 변경, 발표 시점 등 날짜와 일정에 관한 문의",
        "과제": "과제 제출, 채점 기준, 피드백, 재제출에 관한 문의",
        "기술": "로그인 실패, 영상 재생 오류, 에러, 서버 장애 등 기술 문제 문의",
        "기타": "위 어디에도 속하지 않는 일반 문의",
    }
    model = SentenceTransformer(cfg["embed_model"])
    lab_vec = model.encode(
        [label_desc[l] for l in LABELS], normalize_embeddings=True
    )
    txt_vec = model.encode([r["text"] for r in rows], normalize_embeddings=True)
    sims = txt_vec @ lab_vec.T  # 정규화했으므로 내적이 곧 코사인 유사도
    out = []
    for row_sims in sims:
        j = int(row_sims.argmax())
        out.append((LABELS[j], float(row_sims[j]), dict(zip(LABELS, row_sims.tolist()))))
    return out


CANDIDATES = {
    "baseline_keyword": predict_baseline,
    "tfidf_logreg": predict_tfidf,
    "embedding_siglip_like": predict_embedding,
}


def evaluate(rows, preds, threshold):
    n = len(rows)
    wrong, escalated, correct, out_rows = [], [], 0, []
    for r, (label, score, detail) in zip(rows, preds):
        low_conf = score < threshold
        final = "사람에게 넘김" if low_conf else label
        ok = (not low_conf) and label == r["gold_label"]
        if low_conf:
            escalated.append(r["id"])
        elif not ok:
            wrong.append({"id": r["id"], "text": r["text"], "gold": r["gold_label"], "pred": label, "score": score})
        correct += ok
        r_out = dict(r)
        r_out.update(pred=label, score=score, final=final, detail=detail)
        out_rows.append(r_out)
    auto = n - len(escalated)
    return out_rows, {
        "총건수": n,
        "자동처리": auto,
        "사람에게넘김": len(escalated),
        "자동처리중오분류": len(wrong),
        "오분류율(자동처리기준)": (len(wrong) / auto) if auto else None,
        "정답률(전체기준)": correct / n,
        "오분류목록": wrong,
        "넘긴건_id": escalated,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="tickets.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="이 값 아래 점수는 사람에게 넘긴다")
    p.add_argument("--tfidf-C", type=float, default=1.0)
    p.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--candidates", default=",".join(CANDIDATES))
    args = p.parse_args()

    cfg = {"tfidf_C": args.tfidf_C, "embed_model": args.embed_model}
    rows = load(args.data)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name in args.candidates.split(","):
        name = name.strip()
        print(f"\n=== {name} (threshold={args.threshold}) ===")
        try:
            preds = CANDIDATES[name](rows, cfg)
        except Exception as e:
            print(f"  실행 실패: {type(e).__name__}: {e}")
            summary[name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        out_rows, metrics = evaluate(rows, preds, args.threshold)
        summary[name] = metrics
        print(f"  자동처리 {metrics['자동처리']}/{metrics['총건수']}, "
              f"오분류 {metrics['자동처리중오분류']}건, "
              f"사람에게 넘김 {metrics['사람에게넘김']}건")
        for w in metrics["오분류목록"]:
            print(f"    [오분류] id={w['id']} 정답={w['gold']} 예측={w['pred']} 점수={w['score']}")

        # 중간 산출물: 건별 점수를 그대로(반올림 없이) 남긴다. 덮어쓰지 않는다.
        stem = f"{name}__thr{args.threshold}"
        detail_path = _nonclobber(outdir / f"{stem}.jsonl")
        with open(detail_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 눈으로 확인할 형태
        report_path = _nonclobber(outdir / f"{stem}.md")
        _write_report(report_path, name, args, rows, out_rows, metrics)
        print(f"  → {detail_path.name}, {report_path.name}")

    sum_path = _nonclobber(outdir / f"summary__thr{args.threshold}.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 → {sum_path}")


def _nonclobber(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while (cand := path.with_name(f"{path.stem}__{i}{path.suffix}")).exists():
        i += 1
    return cand


def _write_report(path, name, args, rows, out_rows, metrics):
    lines = [
        f"# {name}",
        "",
        f"- 임계값: {args.threshold}",
        f"- 모델/설정: embed_model={args.embed_model}, tfidf_C={args.tfidf_C}",
        f"- 자료: {args.data} ({len(rows)}건)",
        "",
        "| id | 문의 | 정답 | 예측 | 점수 | 최종 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in out_rows:
        mark = "" if r["pred"] == r["gold_label"] else " ⚠"
        lines.append(
            f"| {r['id']} | {r['text']} | {r['gold_label']} | {r['pred']}{mark} | {r['score']} | {r['final']} |"
        )
    lines += ["", "## 집계", "```json",
              json.dumps({k: v for k, v in metrics.items() if k != "오분류목록"},
                         ensure_ascii=False, indent=2),
              "```"]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
