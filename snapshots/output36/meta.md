# output36 — Qwen3-32B QLoRA (train+dev 통합, Epoch 2)

## 점수
**52.2273** (ROUGE-1: 0.6075, ROUGE-2: 0.4219, ROUGE-L: 0.5374) ← output35(54.0594) 대비 퇴보 (-1.83)

## 설정
- 모델: Qwen/Qwen3-32B
- 방법: 4-bit NF4 QLoRA
- LoRA: r=32, alpha=64, dropout=0.05, target_modules 7개
- 학습 데이터: train+dev 통합 (12,956건)
- 학습: lr=2e-4, epoch=2, batch=1×GA16=16
- 추론: beam=4, length_penalty=0.7, max_new_tokens=100
- enable_thinking=False (Qwen3 thinking 모드 비활성)

## output35와의 차이
- 모델: Qwen3-14B → **Qwen3-32B**
- 데이터/하이퍼파라미터: 동일

## Epoch별 ROUGE (val 100샘플)
[Epoch 1] ROUGE-1: 0.3527 | ROUGE-2: 0.1747 | ROUGE-L: 0.3423 | avg: 0.8696
[Epoch 2] ROUGE-1: 0.4749 | ROUGE-2: 0.3121 | ROUGE-L: 0.4649 | avg: 1.2519

## 예측 요약 길이
- 평균: 83.6자 (ref avg: 81.2자)

## 생성일
2026-05-08 07:23
