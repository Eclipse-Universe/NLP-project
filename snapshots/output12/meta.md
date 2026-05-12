# output12 — Qwen v3 r=32 (단독 모델 최고)

## 점수
ROUGE-1: 0.6164 | ROUGE-2: 0.4406 | ROUGE-L: 0.5519 | **final_result: 53.3054**

## 설정
- 모델: Qwen/Qwen2.5-7B-Instruct
- config: config/config_qwen_v3.yaml
- 체크포인트: outputs/checkpoints_qwen_v3/best_rouge_checkpoint (epoch2, dev ROUGE avg 미기록)
- LoRA: r=32, lora_alpha=64, lr=2e-4, 3 epochs
- 추론: beam=4, lp=0.7, max_new_tokens=100
- 예측 길이: avg 83.8자

## 재현 명령
```bash
python src/inference_lora.py \
    --config config/config_qwen_v3.yaml \
    --checkpoint outputs/checkpoints_qwen_v3/best_rouge_checkpoint \
    --output_name output12.csv
```

## git commit: f453a08
