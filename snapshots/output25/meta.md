# output25 — output24 + output21 앙상블

## 점수
미제출 (제출 후 업데이트 예정)

## 설정
- 앙상블 구성: output24 (bias=0) + output21 (bias=3)
- output24 우선, output21은 3자 이상 가까울 때만 교체
- 선택 비율: output24 207건(41.5%), output21 292건(58.5%)
- 평균 길이: 82.2자

## 앙상블 명령
```bash
python src/ensemble.py \
  --inputs outputs/predictions/output24.csv outputs/predictions/output21.csv \
  --priorities 0 3 \
  --output outputs/predictions/output25.csv
```

## 기대 효과
- v6(beam 콜백 수정, epoch3 best ROUGE 0.7580)이 41.5% 기여
- output21(53.4659)에서 개선 기대: 목표 53.5+
- output24가 직접 제출됐으면 단독 성능 파악 가능했으나 앙상블 전략으로 바로 결합

## 제출 결과
미제출 — 점수 확인 후 업데이트
