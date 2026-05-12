# output30 — Qwen2.5-14B 4-bit QLoRA (Epoch 2 best)

## 점수
**53.5031** (ROUGE-1: 0.6185, ROUGE-2: 0.4408, ROUGE-L: 0.5458) ← 현재 최고

## 설정
- 모델: Qwen/Qwen2.5-14B-Instruct
- 방법: 4-bit NF4 QLoRA (BitsAndBytesConfig, double quant)
- LoRA: r=32, alpha=64, dropout=0.05, target_modules 7개
- 학습: lr=2e-4, epoch=3, batch=1×GA16=16, paged_adamw_8bit
- max_length: 768 (OOM 방지, 99%ile=611토큰)
- rouge_eval_samples: 100 (OOM 방지)
- 추론: beam=4, length_penalty=0.7, max_new_tokens=100

## Epoch별 ROUGE (val 100샘플, beam=4)
| Epoch | ROUGE-1 | ROUGE-2 | ROUGE-L | avg |
|-------|---------|---------|---------|-----|
| 1 | 0.3024 | 0.1268 | 0.2896 | 0.7188 |
| **2** | **0.3150** | **0.1354** | **0.3015** | **0.7519 ← Best** |
| 3 | 0.3046 | 0.1285 | 0.2944 | 0.7276 |

- Epoch 2가 best (Epoch 3 과적합, 7B와 동일 패턴)
- best_rouge_checkpoint: /root/outputs/checkpoints_qwen14b/best_rouge_checkpoint

## 예측 요약 길이
- 평균: 82.5자 | 최대: 181 | 최소: 21 (ref avg: 81.2자)

## 재현
```bash
python src/train_lora.py --config config/config_qwen14b.yaml
python src/inference_lora.py \
  --config config/config_qwen14b.yaml \
  --checkpoint /root/outputs/checkpoints_qwen14b/best_rouge_checkpoint \
  --output_name output30.csv
```

## git commit
5de7e33
