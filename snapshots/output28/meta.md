# output28 — MBR Decoding (v3 checkpoint, N=10)

## 점수
| 지표 | 값 |
|---|---|
| ROUGE-1 | 0.5935 |
| ROUGE-2 | 0.4091 |
| ROUGE-L | 0.5255 |
| **final_result** | **50.9351** |

output12(53.30) 대비 **-2.37점** — MBR 역효과

## 설정
- 모델: Qwen v3 best_rouge_checkpoint (재학습 없음, output12와 동일 checkpoint)
- 방법: MBR (Minimum Bayes Risk) Decoding, N=10 샘플
- temperature=0.8, top_p=0.9
- 참조: "Follow the Wisdom of the Crowd" (ACL Findings 2023)

## 방법론
1. 각 dialogue당 temperature 샘플링으로 10개 후보 생성
2. 후보 간 pairwise ROUGE 계산 (R1+R2+RL 평균)
3. 평균 ROUGE가 가장 높은 후보 선택 ("합의 요약")

## 결과
- 길이: avg **82.1자** | max 187 | min 22
- ref 평균(81.2자)에 beam=4(83.8자)보다 더 근접
- 후보 유효수: 10.0/10 (전 샘플 완전 유효)
- 소요 시간: ~40분 (N=10이 단일 포워드 패스로 처리됨)

## 실패 원인
1. temp=0.8 샘플들이 beam=4보다 개별 품질 낮음
2. N=10은 통계적으로 불충분 (논문은 N=100+ 사용)
3. beam=4가 이 태스크에서 이미 최적 탐색 방식 → 샘플링은 노이즈

## 재현 명령
```bash
python src/inference_mbr.py \
  --config config/config_qwen_v3.yaml \
  --checkpoint outputs/checkpoints_qwen_v3/best_rouge_checkpoint \
  --output_name output28_mbr_n10.csv \
  --n_samples 10 --temperature 0.8 --top_p 0.9
```
