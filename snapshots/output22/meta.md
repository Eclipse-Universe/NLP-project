# output22 — Qwen v3 + min_new_tokens=25

## 점수
| 지표 | 값 |
|---|---|
| ROUGE-1 | 0.6074 |
| ROUGE-2 | 0.4328 |
| ROUGE-L | 0.5417 |
| **final_result** | **52.7315** |

## 설정
- 모델: Qwen2.5-7B-Instruct + LoRA r=32 (v3 체크포인트 재사용)
- 체크포인트: `/root/outputs/checkpoints_qwen_v3/best_rouge_checkpoint`
- 변경사항: `min_new_tokens: 25` 추가 (그 외 동일)
- beam=4, length_penalty=0.7, min_new_tokens=25

## 결과 분석
- 예측 길이: avg **85.6자** | max 177 | min 36
- output12(avg 83.8자) 대비 더 길어짐 → ref 81.2자에서 멀어짐
- output12(53.3054) 대비 **-0.57점 퇴보**
- 원인: min_new_tokens가 beam search 강제 연장 → 필요 이상으로 긴 요약 생성

## 교훈
- **min_new_tokens 금지**: 이 태스크에서 beam이 자연스럽게 최적 길이 선택 (min 없이 avg ~84자)
- 너무 짧은 요약이 문제라면 length_penalty를 높이는 것이 더 자연스러운 접근

## 재현 명령
```bash
# config_qwen_v3.yaml에 min_new_tokens: 25 임시 추가 후
python src/inference_lora.py \
  --config config/config_qwen_v3.yaml \
  --checkpoint outputs/checkpoints_qwen_v3/best_rouge_checkpoint \
  --output_name output22.csv
```

## git hash
b533a91 (output22/23 생성 + v6 OOM 수정 재시작)
