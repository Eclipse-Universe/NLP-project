# output21 — output16 + output20 앙상블 (현재 최고)

## 점수
ROUGE-1: 0.6152 | ROUGE-2: 0.4405 | ROUGE-L: 0.5482 | **final_result: 53.4659**
(output16 53.39 대비 +0.08점, 미미한 개선)

## 설정
- 앙상블: output16(80.4%, 401건) + output20(19.6%, 98건)
- 선택 기준: |len - 81.2자| 최소, output16 bias=3자 우선
- 예측 avg 81.3자 (reference 81.2자에 가장 근접)

## 앙상블 이득이 미미한 원인
- output20 단독 52.54 (v3보다 낮음) → 좋은 샘플 19.6%만 기여
- 길이 기반 앙상블 = content-agnostic (내용 품질 반영 안 됨)
- 앙상블 파트너가 강해야 전체 이득이 큼

## git commit: 5d7b37d
