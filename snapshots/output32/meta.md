# output32 — output30 + output21 앙상블 (역방향)

## 점수
**53.2959** (ROUGE-1: 0.6147, ROUGE-2: 0.4384, ROUGE-L: 0.5457) — output30 단독보다 퇴보

## 퇴보 원인
- output21(7B 앙상블)이 58.7%를 차지하며 output30(14B) 성능을 희석
- 길이 기반 선택은 의미 품질을 반영 못함: output21이 길이상 ref에 더 가까운 케이스가 많음
- output30(82.5자 avg)이 output21(81.3자 avg)보다 ref(81.2자)에서 멀어 불리한 선택 구조

## 배경
output31의 교훈: 강한 모델(output30)에 낮은 bias, 약한 모델(output21)에 높은 bias 부여

## 설정
- 앙상블 방식: 길이 기반 샘플별 선택 (|len - 81.2|)
- output30: bias=0 (우선, 단독 최고 53.5031)
- output21: bias=3 (보조, 53.4659)

## 앙상블 결과
- output30 선택: 206건 (41.3%)
- output21 선택: 293건 (58.7%)
- 결과 평균 길이: 80.7자

## 재현
```bash
python src/ensemble.py \
  --inputs outputs/predictions/output30.csv outputs/predictions/output21.csv \
  --priorities 0 3 \
  --output outputs/predictions/output32.csv
```

## git commit
885a02f
