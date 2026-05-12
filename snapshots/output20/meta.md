# output20 — Qwen v5 (train+dev 12956건, r=32, epoch2)

## 점수
ROUGE-1: 0.6066 | ROUGE-2: 0.4289 | ROUGE-L: 0.5407 | **final_result: 52.5391**
(v3 단독 53.30보다 -0.76점 퇴보)

## 설정
- config: config/config_qwen_v5.yaml
- 체크포인트: outputs/checkpoints_qwen_v5/checkpoint-1620 (epoch2 수동 선택)
- 학습 데이터: train(12457) + dev(499) = 12,956건, LoRA r=32, lr=2e-4
- 추론: beam=4, lp=0.7, 예측 avg 84.2자

## 실패 원인
- data leakage: dev_holdout 50건이 train_all에 포함 → ROUGE callback 신뢰 불가
- RougeEvalCallback greedy 평가 → beam=4 추론과 checkpoint 선택 기준 불일치

## git commit: 5d7b37d
