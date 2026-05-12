# output18 — Qwen v4 + Qwen v3 앙상블

## 점수
ROUGE-1: 0.6097 | ROUGE-2: 0.4333 | ROUGE-L: 0.5425 | **final_result: 52.8486**
(output12 53.30보다 낮음 — v4 혼합이 오히려 역효과)

## 설정
- 앙상블: output12(Qwen v3) 63.7% + output17(Qwen v4) 36.3%
- 선택 기준: 샘플별 |len - 81.2자| 최소, output12 bias=3자 우선
- 예측 길이: avg 80.3자

## 교훈
- v4 예측의 품질이 낮아서 v3와 섞어도 전체 성능 하락
- 앙상블 파트너는 단독으로도 v3와 비슷한 품질이어야 효과 있음

## git commit: b0fcb1f
