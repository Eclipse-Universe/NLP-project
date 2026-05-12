# output19 — 3-way 앙상블 (v3 + v3+EXAONE + v4)

## 점수
ROUGE-1: 0.6135 | ROUGE-2: 0.4387 | ROUGE-L: 0.5474 | **final_result: 53.3206**
(output16 53.39보다 미세하게 낮음)

## 설정
- 앙상블: output12(229건, 45.9%) + output16(156건, 31.3%) + output17(114건, 22.8%)
- 선택 기준: 샘플별 |len - 81.2자| 최소, output12 bias=3자 우선
- 예측 길이: avg 80.3자

## 교훈
- 3-way지만 실질적으로 output12가 46% 지배 → output16과 유사한 결과
- v4를 22% 포함했지만 output16(v3+EXAONE)보다 낮음

## git commit: b0fcb1f
