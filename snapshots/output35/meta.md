# output35 — Qwen3-14B QLoRA (train+dev 통합, Epoch 2)

## 점수
54.0594 (ROUGE-1: 0.6240, ROUGE-2: 0.4423, ROUGE-L: 0.5554) ← 현재 최고

## 설정
- 모델: Qwen/Qwen3-14B
- 방법: 4-bit NF4 QLoRA
- LoRA: r=32, alpha=64, dropout=0.05, target_modules 7개
- 학습 데이터: train+dev 통합 (12,956건)
- 학습: lr=2e-4, epoch=2, batch=1×GA16=16
- 추론: beam=4, length_penalty=0.7, max_new_tokens=100
- enable_thinking=False (Qwen3 thinking 모드 비활성)

## output34와의 차이
- 모델: Qwen2.5-14B → **Qwen3-14B**
- 데이터/하이퍼파라미터: 동일

## Epoch별 ROUGE (val 100샘플)
[Epoch 1] ROUGE-1: 0.3745 | ROUGE-2: 0.2016 | ROUGE-L: 0.3635 | avg: 0.9397
[Epoch 2] ROUGE-1: 0.4354 | ROUGE-2: 0.2732 | ROUGE-L: 0.4265 | avg: 1.1351

## 예측 요약 길이
- 평균: 82.9자 (ref avg: 81.2자)

## 생성일
2026-05-07 00:49
