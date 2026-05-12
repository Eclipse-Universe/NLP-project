# output34 — Qwen2.5-14B QLoRA (train+dev 통합, Epoch 2)

## 점수
53.9091 (ROUGE-1: 0.6201, ROUGE-2: 0.4436, ROUGE-L: 0.5536) ← 현재 최고

## 설정
- 모델: Qwen/Qwen2.5-14B-Instruct
- 방법: 4-bit NF4 QLoRA (output30과 동일 하이퍼파라미터)
- LoRA: r=32, alpha=64, dropout=0.05, target_modules 7개
- 학습 데이터: train(12,457) + dev(499) 통합 = 12,956건
- 학습: lr=2e-4, epoch=2 (sweet spot 직접 지정), batch=1×GA16=16
- max_length: 768, rouge_eval_samples: 100
- 추론: beam=4, length_penalty=0.7, max_new_tokens=100

## output30과의 차이
- 데이터: train only(12,457) → **train+dev 통합(12,956, +499샘플)**
- epoch: 3 (best_rouge 선택) → **2 (직접 지정)**
- 나머지 모두 동일

## Epoch별 ROUGE (val 100샘플 기준)
| Epoch | ROUGE-1 | ROUGE-2 | ROUGE-L | avg |
|-------|---------|---------|---------|-----|
| 1 | - | - | - | 0.8770 |
| **2** | **0.4265** | **0.2730** | **0.4163** | **1.1158 ← Best** |

## 예측 요약 길이
- 평균: 82.7자 (ref avg: 81.2자)

## 재현
```bash
python src/prepare_traindev.py --data_path /root/data
python src/train_lora.py --config config/config_qwen14b_traindev.yaml
python src/inference_lora.py \
  --config config/config_qwen14b_traindev.yaml \
  --checkpoint /root/outputs/checkpoints_qwen14b_traindev/best_rouge_checkpoint \
  --output_name output34.csv
```

## 생성일
2026-05-06 07:20
