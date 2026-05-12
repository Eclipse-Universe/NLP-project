# output33 — Multi-model MBR 앙상블 (output12 + output14 + output21 + output30)

## 점수
미제출 (제출 대기 중)

## 방법
Multi-model MBR (Minimum Bayes Risk):
- 각 샘플마다 4개 모델 예측 간 쌍방향 ROUGE 계산
- 평균 ROUGE가 가장 높은 예측 선택 (컨센서스 중심)
- 멘토 권장 방법: "데이터 하나하나마다 루지를 본 다음 가장 높은 루지 값을 정답으로"

## 입력 모델
- output12 (Qwen2.5-7B v3, 단독 53.3054)
- output14 (EXAONE-3.5-7.8B, 단독 53.15)
- output21 (Qwen7B 길이기반 앙상블, 53.4659)
- output30 (Qwen2.5-14B QLoRA, 단독 53.5031 ← 최고)

## 앙상블 결과
- output12 선택: 266건 (53.3%)
- output14 선택: 117건 (23.4%)
- output30 선택: 103건 (20.6%)
- output21 선택: 13건 (2.6%)
- 결과 평균 길이: 83.8자 (ref 81.2자 대비 +2.6자)

## 분석
- output21이 2.6%로 낮은 이유: 이미 앙상블된 예측이라 standalone 모델들과 ROUGE 합의 낮음
- output12 우세: 가장 "합의 중심적" 예측 생성 (단, 단독 성능은 output30보다 낮음)
- output30이 20.6%로 선택됨: output30 단독보다 낮을 가능성 있음

## 재현
```bash
python src/mbr_ensemble.py \
  --inputs outputs/predictions/output12.csv \
           outputs/predictions/output14.csv \
           outputs/predictions/output21.csv \
           outputs/predictions/output30.csv \
  --output outputs/predictions/output33.csv
```
