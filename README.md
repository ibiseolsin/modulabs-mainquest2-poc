# modulabs-mainquest2-poc
모두의연구소 AI 에이전트 1기 Main Quest 2 — 내 도메인에서 AI 개선 지점 발굴 및 PoC 구현

## 산출물

- **[REPORT.md](REPORT.md)** — 과제 본문. 후보 3지점 도출 → 선정 → 성공 기준 → 후보 비교 → 에이전트 지시문 → 회고
- [run_poc.py](run_poc.py) — 후보 3개를 같은 자료로 비교하는 PoC 스크립트
- [data/tickets.csv](data/tickets.csv) — 문의 20건. 정답 라벨은 모델을 돌리기 전에 작성
- `results/` — 임계값별 건별 점수(반올림 없음)와 확인용 표. 다시 돌려도 덮어쓰지 않음

## 선정한 지점

수강생 문의 1차 분류·배분 (분류와 배분 유형). 병목은 판단 자체가 아니라 판단의 총량.

성공 기준: **오분류가 100건 중 5건 이하이고, 확신이 낮은 건은 사람에게 넘어간다.**

## 검증 결과

| 후보 | 실행 위치 | 정답 |
| --- | --- | --- |
| 기준선 · 키워드 규칙 | 노트북 | **18/20** |
| 다국어 임베딩 유사도 | 노트북 | 12/20 |
| TF-IDF + 로지스틱 | 노트북 | 4/20 |
| ~~외부 LLM API~~ | ~~외부~~ | 개인정보 조건으로 점수 측정 전 제외 |

**기준선이 이겼습니다.** "AI가 쓸모없다"가 아니라 "이 업무에는 지금 방식이 낫다"는 결과이고, 그것도 PoC의 결과입니다.

성공 기준(≤5%)을 만족하는 임계값은 1.0뿐이며 그때 자동 처리 비율은 55%입니다. 병목이 판단의 총량이었던 점을 감안해 **조건부 통과**로 판정했습니다. 자세한 근거와 한계는 [REPORT.md](REPORT.md)에 있습니다.

## 재현

```bash
pip install scikit-learn numpy sentence-transformers
python run_poc.py --candidates baseline_keyword,tfidf_logreg,embedding_siglip_like --threshold 0.0
python run_poc.py --candidates baseline_keyword --threshold 1.0
```
