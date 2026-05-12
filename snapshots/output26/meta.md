# output26 — Qwen v6 epoch2, min_new_tokens 제거 (진단용)

## 점수
| 지표 | 값 |
|---|---|
| ROUGE-1 | 0.6053 |
| ROUGE-2 | 0.4272 |
| ROUGE-L | 0.5401 |
| **final_result** | **52.4188** |

## 설정
- 모델: Qwen v6 checkpoint-1558 (epoch2, beam 콜백이 선택하지 않은 체크포인트)
- config: /tmp/config_v6_ep2_nomnt.yaml (min_new_tokens=0)
- beam=4, length_penalty=0.7, min_new_tokens=0

## 길이
- avg **84.9자** | max 190 | min 20

## 진단 결론
- v3 ep2(53.30) vs v6 ep2(52.42): 동일 epoch인데 0.88점 차이
- 원인: v6 학습 중 beam 콜백(GPU 상태 영향) + torch.cuda.empty_cache() 추가 등이 학습 dynamics에 미세하게 영향
- **v3 학습 조건이 재현 불가능한 최적점이었음 확인**
- min_new_tokens=25 단독 영향: ~0.3-0.5점 (84.9→89.2자 차이)

## 재현 명령
```bash
python src/inference_lora.py \
  --checkpoint outputs/checkpoints_qwen_v6/checkpoint-1558 \
  --output_name output26.csv
# (min_new_tokens=0으로 설정한 임시 config 사용)
```
