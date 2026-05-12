# output31 — output21 + output30 앙상블

## 점수
**53.1858** (ROUGE-1: 0.6133, ROUGE-2: 0.4370, ROUGE-L: 0.5452) — output30 단독보다 퇴보

## 퇴보 원인 분석
- output30(53.5031) > output21(53.4659)인데, output21에 bias=0, output30에 bias=3 부여
- 결과: 더 강한 output30이 87건(17.4%)만 선택, 약한 output21이 412건(82.6%) 지배
- bias=3 때문에 output30이 억제된 케이스: **112건** — 이 샘플들이 quality 하락 주도
- 교훈: 앙상블 전 단독 성능 확인 후 bias 방향 설정 필수

## 설정
- 앙상블 방식: 길이 기반 샘플별 선택 (|len - 81.2|)
- output21: bias=0 (잘못된 우선순위, 실제로는 약한 소스)
- output30: bias=3 (잘못된 페널티, 실제로는 강한 소스)

## 앙상블 결과
- output21 선택: 412건 (82.6%)
- output30 선택: 87건 (17.4%)
- 결과 평균 길이: 80.8자

## git commit
5de7e33
