# output24 — Qwen v6 (beam 콜백 수정, epoch3 best)

## 점수
미제출 (output25 앙상블용)

## 설정
- 모델: Qwen2.5-7B-Instruct + LoRA r=32 (v6)
- 체크포인트: `/root/outputs/checkpoints_qwen_v6/best_rouge_checkpoint` (epoch3)
- config: `config/config_qwen_v6.yaml`
- beam=4, length_penalty=0.7, min_new_tokens=25

## v6 핵심 변경 (v3 대비)
- **RougeEvalCallback 수정**: greedy → beam=4+lp=0.7 (추론 조건 일치)
- 결과: epoch3이 진짜 best로 선택됨 (v1~v5에서는 항상 epoch2가 sweet spot이었지만 greedy 콜백의 오류)
- epoch별 ROUGE (beam 기반):
  - Epoch 1: avg 0.6912
  - Epoch 2: avg 0.7041
  - Epoch 3: avg **0.7580** ← Best (v3 epoch2 greedy avg ~0.69보다 높음)

## 예측 결과
- 길이: avg **89.2자** | max 191 | min 39
- min_new_tokens=25 영향으로 v3(83.8자)보다 더 길어짐

## 재현 명령
```bash
python src/inference_lora.py \
  --config config/config_qwen_v6.yaml \
  --checkpoint outputs/checkpoints_qwen_v6/best_rouge_checkpoint \
  --output_name output24.csv
```
