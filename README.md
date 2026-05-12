# 한국어 대화 요약 — NLP 대회 프로젝트

Qwen2.5 / Qwen3 + QLoRA fine-tuning으로 한국어 대화를 요약하는 모델을 개발했습니다.  
평가 지표: `final_result = mean(ROUGE-1, ROUGE-2, ROUGE-L) × 100`

---

## 최고 점수

| 구분 | output | 설정 | 점수 |
|---|---|---|---|
| **전체 최고** | **output35** | Qwen3-14B 4-bit QLoRA, train+dev, epoch2, beam=4, lp=0.7 | **54.0594 ← 현재 최고** |
| Qwen2.5 최고 | output34 | Qwen2.5-14B 4-bit QLoRA, train+dev, epoch2 | 53.9091 |
| train-only 최고 | output30 | Qwen2.5-14B 4-bit QLoRA, train only, epoch2 | 53.5031 |
| 7B 단독 최고 | output12 | Qwen2.5-7B LoRA r=32, epoch2, beam=4, lp=0.7 | 53.3054 |

---

## 전체 점수 추이

| 제출 | 모델 | 주요 변경 | 점수 |
|---|---|---|---|
| output1 | KoBART | 베이스라인 (버그 포함) | 18.27 |
| output2 | KoBART | `#PersonN#` 버그 수정, 길이 제한 | 44.55 |
| output3 | Qwen2.5-7B + LoRA r=16 | greedy 추론 | 50.20 |
| output4 | Qwen2.5-7B + LoRA r=16 | beam=4 | 51.25 |
| output7 | Qwen2.5-7B + LoRA r=16 | length_penalty=0.7 | 52.58 |
| output11 | Qwen2.5-7B + LoRA r=32 | lr=1e-4, epoch2 | 52.88 |
| output12 | Qwen2.5-7B + LoRA r=32 | lr=2e-4, epoch2 | **53.30** |
| output14 | EXAONE-3.5-7.8B + LoRA r=32 | weight tying 수정 | 53.15 |
| output16 | Qwen v3 + EXAONE v2 앙상블 | 샘플별 길이 기반 선택 | 53.39 |
| output17 | Qwen2.5-7B + LoRA r=64 | rank 확장 시도 | 51.71 (퇴보) |
| output21 | output16 + output20 앙상블 | 길이 기반 2-way | 53.47 |
| output22 | Qwen2.5-7B + LoRA r=32 | min_new_tokens=25 추가 | 52.73 (퇴보) |
| output23~29 | 각종 앙상블/MBR 시도 | 다양한 조합 | 50.94~53.17 (전부 퇴보) |
| output30 | **Qwen2.5-14B 4-bit QLoRA** | 모델 업그레이드, train only | **53.5031** |
| output31~33 | 앙상블/MBR 재시도 | output30 기반 조합 | 53.19~53.45 (전부 퇴보) |
| output34 | **Qwen2.5-14B 4-bit QLoRA** | train+dev 데이터 통합 | **53.9091** (+0.41) |
| output35 | **Qwen3-14B 4-bit QLoRA** | Qwen3 아키텍처 업그레이드 | **54.0594** (+0.15) |
| output36 | Qwen3-32B 4-bit QLoRA | 모델 확장 시도 (bf16 패치) | 52.2273 (퇴보 -1.83) |

---

## 주요 발견 (실험을 통해 확인된 규칙)

| 발견 | 결과 |
|---|---|
| `skip_special_tokens=True` | `#Person1#` 화자 표기 전부 제거 → **금지** |
| `no_repeat_ngram_size≥3` | `#Person1#` 반복 3-gram 처리 → 생성 조기 종료 → **금지** |
| `min_new_tokens>0` | 강제 길이 증가 → ref 81.2자에서 멀어짐 → **금지** |
| epoch2가 항상 sweet spot | v1~v5 전부 epoch3에서 과적합. Qwen3도 동일 확인 |
| r=32가 최적 LoRA rank | r=16→r=32: +0.72점, r=32→r=64: -1.59점 (style drift) |
| beam=4 + lp=0.7 고정 | greedy 대비 +2.38점 |
| 데이터 확장 효과 큼 | train only → train+dev: **+0.41점** (output30→output34) |
| 모델 업그레이드 효과 | Qwen2.5-14B → Qwen3-14B: **+0.15점** (output34→output35) |
| 앙상블은 단독보다 지속 열위 | 이질 모델 MBR, 길이 기반 앙상블 모두 단독 최고 모델에 패배 |
| Qwen3 thinking 비활성 필수 | `enable_thinking=False` 없으면 `<think>` 태그 출력 오염 |
| 32B 모델 오히려 퇴보 | Qwen3-32B: 52.2273 (14B 대비 -1.83점). 24GB GPU에서 bf16 패치 필요 → 학습 품질 저하 가능성 |

---

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# train+dev 통합 데이터 생성
python src/prepare_traindev.py --data_path data/

# 학습 (현재 최고: Qwen3-14B)
python src/train_lora.py --config config/config_qwen3_14b_traindev.yaml

# 추론
python src/inference_lora.py \
  --config config/config_qwen3_14b_traindev.yaml \
  --checkpoint outputs/checkpoints_qwen3_14b_traindev/best_rouge_checkpoint \
  --output_name output35.csv
```

---

## 프로젝트 구조

```
src/
├── preprocess.py              # 데이터 정제 (\n, <br> 노이즈, 화자 표기 처리)
├── train_lora.py              # QLoRA 학습 (RougeEvalCallback, Qwen3 대응)
├── inference_lora.py          # beam search 추론 (Qwen3 thinking 비활성 포함)
├── prepare_traindev.py        # train+dev 통합 데이터 생성
├── ensemble.py                # 길이 기반 앙상블
└── mbr_ensemble.py            # Multi-model MBR 앙상블

config/
├── config_qwen3_14b_traindev.yaml   # ★ 현재 최고 설정 (output35)
├── config_qwen14b_traindev.yaml     # Qwen2.5-14B train+dev (output34)
├── config_qwen14b.yaml              # Qwen2.5-14B train only (output30)
├── config_qwen_v3.yaml              # Qwen2.5-7B 최고 설정 (output12)
└── ...                              # 실험 이력 config들

snapshots/                     # output별 재현 정보 (점수, config, CSV, meta.md)
progress/                      # 날짜별 실험 기록
notebooks/                     # 외부 실험 노트북 (SimPO, MBR 앙상블)
```

---

## 환경

- GPU: NVIDIA RTX 3090 (24GB)
- CUDA 12.2 / Python 3.10
- 주요 라이브러리: `torch==2.5.1+cu121`, `transformers==5.8.0`, `peft==0.14.0`, `bitsandbytes==0.49.2`
- 학습 시간: Qwen3-14B 기준 약 7.7시간 (2 epochs, 12,956건, RTX 3090)

---

## 확정 하이퍼파라미터 (변경 금지)

| 파라미터 | 값 | 이유 |
|---|---|---|
| `skip_special_tokens` | `False` | True면 #PersonN# 제거 |
| `no_repeat_ngram_size` | `0` | ≥3이면 #Person1# 반복으로 조기 종료 |
| `min_new_tokens` | `0` | 강제 길이 증가 → 성능 하락 |
| `num_beams` | `4` | greedy 대비 +2.38점 |
| `length_penalty` | `0.7` | 최적값 |
| `lora_r` | `32` | r=16: -0.72, r=64: -1.59 |
| `lr` | `2e-4` | r=32에서 최적 |
| `epochs` | `2` | 항상 sweet spot |
