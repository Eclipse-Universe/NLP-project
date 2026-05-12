# output17 — Qwen v4 r=64 (실패, 분석용)

## 점수
ROUGE-1: 0.6000 | ROUGE-2: 0.4195 | ROUGE-L: 0.5317 | **final_result: 51.7074** (v3 대비 -1.59점)

## 설정
- 모델: Qwen/Qwen2.5-7B-Instruct
- config: config/config_qwen_v4.yaml
- 체크포인트: outputs/checkpoints_qwen_v4/best_rouge_checkpoint (epoch2, dev ROUGE avg=0.6939)
- LoRA: r=64, lora_alpha=128, lr=2e-4, 3 epochs
- 추론: beam=4, lp=0.7, max_new_tokens=100
- 예측 길이: avg 81.8자

## 실패 원인 분석
- r=64는 r=32 대비 파라미터 2배 → 기반 모델 instruction following 스타일을 과도하게 변형
- v3(r=32): "화자 행동 + 결과" 간결 포맷 학습 → 레퍼런스와 높은 ROUGE
- v4(r=64): "맥락 + 세부사항" 서술 포맷 학습 → 더 정보량 많지만 레퍼런스와 낮은 ROUGE
- r 스케일링 법칙 붕괴: r=16→32(+0.72점), r=32→64(-1.59점). r=32가 sweet spot

## git commit: b0fcb1f
