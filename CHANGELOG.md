# CHANGELOG — output별 코드 변경 이력

각 output 제출 시점의 코드/설정 변경사항을 기록합니다.
형식: 추가(+) / 수정(M) / 삭제(-) / 변경없음(=)

---

## output1 — KoBART 베이스라인 (18.27점)

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `src/train.py` | KoBART 학습 코드 |
| + | `src/dataset.py` | DialogueDataset 클래스, 토크나이징 |
| + | `src/inference.py` | KoBART 추론, skip_special_tokens=True (버그) |
| + | `src/evaluate.py` | ROUGE 평가 유틸리티 |
| + | `src/preprocess.py` | dialogue 노이즈 정제 (\\n, \<br\> 처리) |
| + | `config/config.yaml` | KoBART 설정 |

### 핵심 버그 (→ 18.27점 원인)
- `inference.py`: `skip_special_tokens=True` → `#Person1#` 등 화자 표기 전부 제거
- `inference.py`: `max_new_tokens` 미설정 → 평균 235.9자 (정답 81.2자의 3배)
- 모델 `digit82/kobart-summarization`: 뉴스 요약 편향 → 장문 생성

---

## output2 — KoBART 버그 수정 (44.55점, +26.28)

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `src/inference.py` | `skip_special_tokens=False` + BOS/EOS/PAD 수동 제거 |
| M | `src/inference.py` | postprocess 추가: `re.sub(r'(#\w+#) (?!#)', r'\1', text)` → `#Person1# 은` → `#Person1#은` |
| M | `src/inference.py` | `max_new_tokens=30`, `length_penalty=0.5` (정답 avg 31토큰에 맞춤) |
| = | 나머지 전체 | 변경 없음 |

---

## output3 — Qwen2.5-7B-Instruct + LoRA 도입 (50.20점, +5.65)

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `src/train_lora.py` | Qwen LoRA 학습 코드 전체 |
| + | `src/inference_lora.py` | Qwen LoRA 추론 코드 |
| + | `config/config_qwen.yaml` | Qwen v1 설정 (r=16, lr=2e-4, 3 epochs, greedy) |

### train_lora.py 핵심 설계
- `build_prompt()`: `apply_chat_template`으로 Qwen 공식 포맷 적용, 학습/추론 겸용
- `make_dataset()`: labels에 `-100` 마스킹 (prompt 부분 loss 제외)
- `CausalLMCollator`: right-padding, 패딩 토큰도 `-100` 처리
- `RougeEvalCallback`: epoch 종료 시 dev 100건 샘플 ROUGE 측정 (wandb 기록용)
- 추론: `greedy`, `num_beams=1`, `max_new_tokens=100`

---

## output4, output5 — Beam Search 실험 (51.25 / 50.75점)

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `config/config_qwen.yaml` | `num_beams: 1 → 4` |
| M | `src/inference_lora.py` | `num_beams=ic['num_beams']` 파라미터화 |
| = | `src/train_lora.py` | 변경 없음 (v1 체크포인트 그대로 사용) |

- output4: epoch2 체크포인트 (eval_loss best) → 51.25점
- output5: epoch3 체크포인트 (ROUGE dev best) → 50.75점
- **결론**: eval_loss best(epoch2)가 ROUGE dev best(epoch3)보다 실제 성능 우수

---

## output6 — length_penalty=0.9 실험 (미제출)

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `config/config_qwen.yaml` | `length_penalty: 0.9` 추가 |
| M | `src/inference_lora.py` | `length_penalty=ic.get('length_penalty', 1.0)` 추가 |

- 평균 90.1자 (정답 81.2자보다 여전히 길어 미제출)

---

## output7 — length_penalty=0.7 (52.58점, +1.33) ★ 전 최고

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `config/config_qwen.yaml` | `length_penalty: 0.9 → 0.7` |
| = | `src/inference_lora.py` | 변경 없음 |

- 평균 84.7자 → 정답 81.2자에 근접, ROUGE precision 개선

---

## output8 — no_repeat_ngram_size=3 실험 (미제출, 포트폴리오용 실패 기록)

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `src/inference_lora.py` | `no_repeat_ngram_size=ic.get('no_repeat_ngram_size', 0)` 추가 |
| + | `/tmp/config_output8.yaml` | `no_repeat_ngram_size: 3` 임시 config (git 미포함) |

- 평균 62.8자 (최소 11자) → 완전 실패
- 원인: `#Person1#`, `#Person2#`가 반복 3-gram으로 처리되어 생성 조기 종료
- **교훈: 한국어 화자 표기 태스크에서 no_repeat_ngram_size 금지**

---

## output9 — Qwen v2 학습 결과 (50.67점, 퇴보)

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `config/config_qwen_v2.yaml` | r=32, lr=1e-4, 5 epochs 목표 |

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `src/train_lora.py` | `RougeEvalCallback`: dev 100건 → **전체 499건**으로 변경 |
| M | `src/train_lora.py` | `RougeEvalCallback`: best ROUGE 갱신 시 `best_rouge_checkpoint/`에 직접 저장 |
| M | `config/config_qwen_v2.yaml` | `load_best_model_at_end: false` (ROUGE callback이 직접 처리) |

- v2 학습: epoch3에서 크래시 (5 중 3 완료), lr=1e-4 너무 낮음
- epoch2 sweet spot 이후 epoch3에서 과적합 → 퇴보

---

## output10 — v2 epoch1 추론 (미제출)
- 코드 변경 없음, checkpoint-779(epoch1) 사용
- 평균 85.5자

---

## output11 — v2 epoch2 추론 (52.88점, +0.30) ★ 전 최고

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| = | 전체 | 변경 없음 |

- checkpoint-1558(epoch2) 사용, config_qwen_v2.yaml
- r=32 효과 확인: 52.58 → 52.88점

---

## output12 — Qwen v3 학습 결과 (53.30점, +0.42) ★ 현재 최고

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `config/config_qwen_v3.yaml` | r=32, **lr=2e-4** (v2 1e-4 복원), 3 epochs |

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| = | `src/train_lora.py` | 변경 없음 |
| = | `src/inference_lora.py` | 변경 없음 |

- v3 epoch2 best_rouge_checkpoint 사용
- **r=32 + lr=2e-4 조합이 현재 최적**

---

## output13 (예정) — EXAONE-3.5-7.8B 학습 결과

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `config/config_exaone.yaml` | EXAONE 전용 target_modules (out_proj, c_fc_0/1, c_proj) |

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `src/train_lora.py` | PEFT 호환성 패치: `get_input_embeddings` 미구현 모델 자동 탐색 패치 |
| M | `src/train_lora.py` | gradient_checkpointing: `enable_input_require_grads()` try/except 추가 |
| M | `src/train_lora.py` | `_REMOVE`에 EXAONE 특수토큰 추가 (`[|endofturn|]` 등) |
| M | `src/inference_lora.py` | `_REMOVE_TOKENS`에 EXAONE 특수토큰 추가 |

- 학습 중 (약 5~6시간 소요)
- 결과 도출 후 이 파일 업데이트 예정

---

## output16 — Qwen v3 + EXAONE v2 앙상블 (제출 예정)

### 앙상블 로직
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `outputs/predictions/output16.csv` | 샘플별 길이 기반 선택 (bias=3자, output12 우선) |

- output12(83.8자) vs output14(84.4자) → output16(평균 81.9자)
- output12에서 295건(59.1%), output14에서 204건(40.9%) 선택
- reference 평균 81.2자에 가장 근접
- 제출 결과: 제출 후 업데이트 필요

---

## output17 — Qwen r=64 (v4) 학습 결과 (51.7074점, 퇴보 -1.59점)

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `config/config_qwen_v4.yaml` | r=64, lora_alpha=128, lr=2e-4, 3 epochs |

- trainable params: 161,480,704 (2.08%) — v3(83.9M) 대비 2배

### Epoch별 dev ROUGE
[Epoch 1] avg: 0.6717
[Epoch 2] avg: 0.6939 ← Best ROUGE → best_rouge_checkpoint 저장
[Epoch 3] avg: 0.6763 (과적합, train_loss=0.22, eval_loss=0.90)

### output17 추론 결과
- 평균 길이: 81.8자 | ROUGE-1: 0.6000 | ROUGE-2: 0.4195 | ROUGE-L: 0.5317
- **final_result: 51.7074 (v3 대비 -1.59점)**

### 실패 원인
- r=64 과도한 LoRA capacity → 기반 모델 instruction following 스타일 변형
- v3(r=32): "화자 행동+결과" 간결 포맷 → 레퍼런스 high ROUGE
- v4(r=64): "맥락+세부사항" 서술 포맷 → 더 정보량 많지만 낮은 ROUGE
- **결론: r=32가 이 태스크의 sweet spot. r=64 금지**

---

## output18 — Qwen v4 + v3 앙상블 (52.8486점, 퇴보)

- output12(63.7%) + output17(36.3%), bias=3 (v3 우선)
- 평균 길이: 80.3자
- **교훈: 품질 낮은 v4 혼합이 오히려 역효과. 앙상블 파트너는 단독 품질이 중요**

---

## output19 — 3-way 앙상블 (53.3206점)

- output12(45.9%) + output16(31.3%) + output17(22.8%)
- 평균 길이: 80.3자
- output16(53.39)보다 낮음: v4 22% 포함이 미세하게 역효과

---

## output20 예정 — Qwen v5 (train+dev 포함 재학습)

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `config/config_qwen_v5.yaml` | r=32, lr=2e-4, train_all(12956건), dev_holdout(50건) |
| M | `src/train_lora.py` | train_file/val_file config 옵션 추가 |
| + | `data/train_all.csv` | train+dev 통합 (12956건) |
| + | `data/dev_holdout.csv` | ROUGE 콜백용 50건 holdout |

- dev 499건을 학습에 포함 → 총 12,956건 (기존 12,457 + 499)
- r=32, lr=2e-4 (v3 최적 설정 유지)
- 학습 중 (총 2,430 steps, ~6시간)
- 결과 도출 후 업데이트 예정

---

## 핵심 인사이트 요약

| 시점 | 발견 | 적용 방식 |
|---|---|---|
| output1→2 | `skip_special_tokens=True`가 화자 표기 제거 | `False`로 변경 + 수동 postprocess |
| output3 | 7B LLM + LoRA가 124M seq2seq 대비 압도적 | 모델 교체 |
| output4→5 | eval_loss best(epoch2) > ROUGE dev best(epoch3) | checkpoint 선택 기준 변경 |
| output6→7 | length_penalty < 1.0이 요약 길이를 정답에 근접 | lp=0.7 고정 |
| output8 | no_repeat_ngram_size=3이 화자 표기 파괴 | 영구 금지 |
| output9→11 | lr=1e-4가 r=32에서 너무 느림, epoch2 항상 sweet spot | lr=2e-4 복원 |
| output12 | r=32 + lr=2e-4 조합 최적 확인 | v3 표준 설정으로 고정 |

## output14 — EXAONE-3.5-7.8B 재학습 v2 (weight tying 수정 후)

### 수정 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `modeling_exaone.py` | ExaoneModel·ForCausalLM에 get/set_input_embeddings 직접 구현 |
| M | `src/train_lora.py` | lambda 패치 제거, enable_input_require_grads 정상 복원 |
| M | `src/inference_lora.py` | device_map='auto' 복원, greedy fallback 제거 |

### Epoch별 dev ROUGE
[Epoch 1] ROUGE-1: 0.2967 | ROUGE-2: 0.1163 | ROUGE-L: 0.2830 | avg: 0.6961
  → Best ROUGE 갱신 (0.6961), 저장: /root/outputs/checkpoints_exaone/best_rouge_checkpoint
[Epoch 2] ROUGE-1: 0.3055 | ROUGE-2: 0.1225 | ROUGE-L: 0.2917 | avg: 0.7197
  → Best ROUGE 갱신 (0.7197), 저장: /root/outputs/checkpoints_exaone/best_rouge_checkpoint
[Epoch 3] ROUGE-1: 0.2933 | ROUGE-2: 0.1153 | ROUGE-L: 0.2810 | avg: 0.6896

### output14 추론 결과
- 예측 요약 길이: 평균 84.4자 | 최대 204 | 최소 18
- 제출 결과: 제출 후 업데이트 필요

## output15 — EXAONE v2 length_penalty=0.65 (52.9385점, 퇴보)

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| = | `src/inference_lora.py` | 변경 없음 |
| M | `/tmp/config_exaone_lp065.yaml` | `length_penalty: 0.7 → 0.65` (임시 config) |

- 평균 82.9자 → reference 81.2자에 가장 근접했으나 recall 손실이 더 큼
- **EXAONE lp 최적값은 0.7로 확정. 이 이하로 내리면 역효과**
- **EXAONE 실험 종료. 최고점 53.15 (output14). Qwen v3(53.30)이 여전히 최고**

---

## 핵심 인사이트 추가

| 시점 | 발견 | 적용 방식 |
|---|---|---|
| output13→14 | EXAONE weight tying 파괴가 성능 저하 원인 | modeling_exaone.py 직접 수정 |
| output14→15 | lp < 0.7이면 recall 손실 > precision 개선 | lp=0.7 전 모델 고정 |
| output12 vs 14 | dev ROUGE 높아도 test가 낮을 수 있음 (Qwen instruction following 우위) | 태스크별 모델 적합성 검증 필요 |

## output20 — Qwen v5 epoch2 추론 결과

### 결과
- 체크포인트: checkpoint-1620 (epoch2), data leakage로 best_rouge 대신 사용
- 예측 길이: avg 84.2자 | max 178 | min 18
- 제출 결과: 제출 후 업데이트 필요

---

## output21 — output20 + output16 앙상블

- output16(최고점 53.39) 기반 + output20 보완
- 제출 결과: 제출 후 업데이트 필요


---

## output20 — Qwen v5 epoch2 (52.5391점, 퇴보)

- checkpoint-1620(epoch2), 학습 데이터 12956건
- **final_result: 52.5391** (v3 53.30 대비 -0.76점)
- 실패 원인: data leakage + greedy/beam 불일치

## output21 — output16 + output20 앙상블 (53.4659점, 신규 최고)

- output16(80.4%) + output20(19.6%), avg 81.3자
- **final_result: 53.4659** (output16 53.39 대비 +0.08점)
- 앙상블 이득 미미: output20 품질 낮아 기여 제한적

---

## output22 — Qwen v3 + min_new_tokens=25 (52.7315점, 퇴보 -0.57)

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `config/config_qwen_v3.yaml` | `min_new_tokens: 25` 추가 (임시) |
| = | `src/inference_lora.py` | 변경 없음 (이미 `ic.get('min_new_tokens', 0)` 지원) |

- v3 best checkpoint 재사용, 재학습 없음
- 길이: avg 85.6자 (output12 83.8자 → 더 길어짐, ref 81.2자에서 멀어짐)
- **final_result: 52.7315** (output12 53.30 대비 -0.57점)
- 실패 원인: min_new_tokens가 beam search를 강제 연장 → 더 긴 생성 → ROUGE 하락
- **결론: min_new_tokens 금지. beam이 자연스럽게 최적 길이 선택하도록 방임**

---

## output23 — output21 + output22 앙상블 (53.1235점, 퇴보 -0.34)

- output21(96.4%) + output22(3.6%), avg 82.0자
- **final_result: 53.1235** (output21 53.4659 대비 -0.34점)
- 실패 원인: output22 품질(52.73) < output21(53.47) → 낮은 품질 혼합이 역효과
- **교훈: 앙상블 source 단독 성능이 현재 최고보다 낮으면 앙상블해도 하락**

---

## output24 — Qwen v6 beam 콜백 수정 (추론 결과)

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `config/config_qwen_v6.yaml` | r=32, lr=2e-4, RougeEvalCallback beam=4+lp=0.7, min_new_tokens=25 |

### v6 핵심 변경
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| M | `src/train_lora.py` | `RougeEvalCallback`: greedy → beam=4+lp=0.7 (추론 조건 일치) |
| M | `src/train_lora.py` | `torch.cuda.empty_cache()` on_epoch_end 전후 추가 (OOM 방지) |
| M | `src/train_lora.py` | `rouge_eval_samples`: 100→50 (beam 평가 메모리 절감) |

### 학습 결과 (beam 기반 ROUGE 콜백)
- Epoch 1: avg 0.6912
- Epoch 2: avg 0.7041
- Epoch 3: avg **0.7580** ← Best → best_rouge_checkpoint 저장
- **핵심 발견**: beam=4 콜백으로 평가하니 epoch3이 진짜 최고. greedy 콜백 v1~v5의 "epoch2 sweet spot"은 콜백 오류였음

### output24 추론 결과
- 길이: avg 89.2자 | max 191 | min 39
- min_new_tokens=25 영향으로 v3(83.8자)보다 더 길어짐

---

## output25 — output24 + output21 앙상블 (52.1864점, 퇴보)

- output24(bias=0, v6 우선) + output21(bias=3)
- 선택: output24 207건(41.5%), output21 292건(58.5%), avg 82.2자
- **final_result: 52.1864** — output24 품질 열위로 앙상블 역효과

---

## output26 — Qwen v6 epoch2 진단 추론 (52.4188점, 퇴보)

- 체크포인트: checkpoint-1558 (v6 epoch2), min_new_tokens=0
- 길이: avg 84.9자 | **final_result: 52.4188**
- v3 ep2(53.30) vs v6 ep2(52.42): 0.88점 차이 → v6 학습 중 beam 콜백이 GPU 상태에 영향
- **결론: v3 학습 조건이 재현 불가능한 최적점이었음. beam 콜백 실험 전체가 역효과**

---

## output27 — output26 + output21 앙상블 (53.1651점, 퇴보)

- output26(38.5%) + output21(61.5%), avg 81.6자
- **final_result: 53.1651** — Qwen LoRA 앙상블 53.3~53.5 천장 재확인

---

## output28 — MBR Decoding, v3 checkpoint, N=10

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `src/inference_mbr.py` | MBR N=10 샘플링 합의 선택 (ACL Findings 2023) |

### 방법론
- v3 best_rouge_checkpoint 재사용 (재학습 없음)
- temperature=0.8, top_p=0.9로 10개 후보 생성
- 후보 간 pairwise ROUGE(R1+R2+RL) 평균이 최고인 "합의 요약" 선택
- 소요 시간: ~40분 (N=10이 단일 포워드 패스)

### 결과
- 길이: avg **82.1자** | max 187 | min 22 (ref 81.2자, beam 83.8자보다 더 근접)
- 후보 유효수: 10.0/10

---

## output29 — MBR(output28) + output21 앙상블

- output28(bias=0, MBR 우선) + output21(bias=3)
- 선택: output28 205건(41.1%), output21 294건(58.9%)
- 평균 길이: **81.2자** — reference 평균과 정확히 일치 (역대 최근접)
- 제출 결과: **52.0300점** (output28 오염으로 퇴보)

---

## output30 — Qwen2.5-14B 4-bit QLoRA Epoch2 (53.5031점, 신규 최고)

### 변경 파일
| 상태 | 파일 | 변경 내용 |
|---|---|---|
| + | `config/config_qwen14b.yaml` | 14B QLoRA 전용 설정 |
| M | `src/train_lora.py` | `use_4bit` 분기, `BitsAndBytesConfig`, `prepare_model_for_kbit_training` 추가 |
| M | `src/inference_lora.py` | `use_4bit` 분기, 4-bit 베이스 모델 로드 지원 |

### 핵심 설정
- 모델: Qwen/Qwen2.5-14B-Instruct (7B → 14B 업그레이드)
- 4-bit NF4 QLoRA: `bnb_4bit_quant_type='nf4'`, `bnb_4bit_use_double_quant=True`
- `prepare_model_for_kbit_training()`: 4-bit 모델 gradient 흐름 활성화
- `optim: paged_adamw_8bit`: QLoRA 권장 옵티마이저
- `max_length: 768` (OOM 방지, 1024→768, 99%ile=611토큰)
- `rouge_eval_samples: 100` (OOM 방지, 499→100)

### Epoch별 ROUGE (val 100샘플, beam=4)
| Epoch | ROUGE-1 | ROUGE-2 | ROUGE-L | avg |
|-------|---------|---------|---------|-----|
| 1 | 0.3024 | 0.1268 | 0.2896 | 0.7188 |
| **2** | **0.3150** | **0.1354** | **0.3015** | **0.7519 ← best** |
| 3 | 0.3046 | 0.1285 | 0.2944 | 0.7276 |

- Epoch 2 best (Epoch 3 과적합, 7B와 동일 패턴)
- 예측 길이: avg 82.5자

---

## output31 — output21 + output30 앙상블 (53.1858점, 퇴보)

- output21(bias=0) + output30(bias=3, 14B QLoRA Epoch2)
- 선택: output21 412건(82.6%), output30 87건(17.4%)
- 평균 길이: 80.8자
- **퇴보 원인**: output30(53.50) > output21(53.47)인데 강한 모델에 bias=3 페널티 → 112건 억제
- 교훈: 앙상블 전 단독 성능 반드시 확인, 강한 모델이 낮은 bias를 가져야 함

---

## output32 — output30(bias=0) + output21(bias=3) 역방향 앙상블 (53.2959점, 퇴보)

- output31 bias 방향 수정: 강한 모델(output30)에 bias=0, 약한 모델(output21)에 bias=3
- 선택: output30 206건(41.3%), output21 293건(58.7%)
- 평균 길이: 80.7자
- **퇴보 원인**: output21(81.3자)이 output30(82.5자)보다 ref(81.2자)에 구조적으로 가까워 길이 기반 선택이 output21을 우대 → 더 강한 14B 예측이 희석됨
- **결론**: output30과의 길이 기반 앙상블은 구조적 한계. output30 단독이 최선.

---

## output33 — Multi-model MBR 앙상블 (점수 미확인, 제출 대기)

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `src/mbr_ensemble.py` | 샘플별 쌍방향 ROUGE 계산 후 평균 최고 예측 선택 |

### 설정
- 입력: output12 (Qwen7B v3), output14 (EXAONE), output21 (7B 앙상블), output30 (Qwen14B)
- 방법: 각 샘플마다 4개 후보 간 ROUGE-1/2/L 평균 계산 → 최고 합의 예측 선택
- 멘토 권장 방식: "데이터 하나하나마다 루지를 본 다음 가장 높은 루지 값을 정답으로"
- 선택 결과: output12 266건(53.3%), output14 117건(23.4%), output30 103건(20.6%), output21 13건(2.6%)
- 결과 평균 길이: 83.8자

---

## 핵심 인사이트 추가

| 시점 | 발견 | 적용 방식 |
|---|---|---|
| output17 | r=64 style drift로 -1.59점 | r=32 고정 (sweet spot) |
| output22 | min_new_tokens=25 → 강제 길이 증가 → 역효과 | min_new_tokens=0 유지 |
| output23 | 낮은 품질 source 앙상블 → 전체 하락 | source 단독 성능 > 현재 최고 필수 |
| output24 | beam 콜백 50샘플 과적합 + min_new_tokens 이중 실패 | greedy 499건 콜백이 최선 |
| output26 | v6 ep2도 v3 ep2보다 0.88점 열위 | beam 콜백 학습 자체가 GPU 상태에 영향 |
| output28 | MBR이 beam보다 ref에 더 가까운 길이 생성(82.1 vs 83.8자) | 합의 선택이 길이 편향 교정 |

---

## output34 — Qwen2.5-14B QLoRA train+dev 통합 (53.9091점, +0.41 신기록)

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `src/prepare_traindev.py` | train.csv + dev.csv → traindev.csv 통합 스크립트 |
| + | `config/config_qwen14b_traindev.yaml` | train+dev 통합 학습 설정 (epoch=2 직접 지정) |

### 코드 변경
| 상태 | 파일 | 내용 |
|---|---|---|
| = | `src/train_lora.py` | 변경 없음 |
| = | `src/inference_lora.py` | 변경 없음 |

### 설정
- output30과 동일 하이퍼파라미터 (r=32, lr=2e-4, beam=4, lp=0.7)
- 학습 데이터: train(12,457) + dev(499) = 12,956건
- epoch=3(best_rouge 선택) → epoch=2 직접 지정

### 교훈
- 데이터 확장이 가장 안정적인 성능 향상 방법 (+0.41점)
- MBR/앙상블보다 효과적

---

## output35 — Qwen3-14B QLoRA train+dev 통합 (54.0594점, +0.15 신기록)

### 신규 파일
| 상태 | 파일 | 내용 |
|---|---|---|
| + | `config/config_qwen3_14b_traindev.yaml` | Qwen3-14B 학습 설정 |

### 코드 변경
| 상태 | 파일 | 내용 |
|---|---|---|
| M | `src/train_lora.py` | Qwen3 enable_thinking=False 대응, \<think\> 태그 제거, set_submodule 패치 |
| M | `src/inference_lora.py` | 동일 Qwen3 대응 패치 |

### 설정
- output34와 동일 하이퍼파라미터
- 모델: Qwen2.5-14B → Qwen3-14B
- torch 2.5.1+cu121, transformers 5.8.0 (Qwen3 지원)

### 환경 이슈
- torch 2.11.0+cu130 → CUDA 12.2 미지원 → 2.5.1+cu121 재설치
- transformers 5.x의 set_submodule (torch 2.6+ 전용) → nn.Module monkey-patch로 해결

### 교훈
- 모델 아키텍처 업그레이드 효과 (+0.15점)
- Qwen3의 thinking 모드는 반드시 비활성화 (enable_thinking=False)
- 데이터 확장 > 모델 업그레이드 순으로 효과적

---

## 최신 핵심 인사이트 (output34~35)

| 시점 | 발견 | 적용 방식 |
|---|---|---|
| output33 | Multi-model MBR: 이질 모델 간 합의 → 강한 모델 페널티 | 동질 모델 MBR만 유효 |
| output34 | train+dev 데이터 통합 → +0.41점 | 데이터 확장이 최고 전략 |
| output35 | Qwen3-14B → +0.15점 추가 향상 | 아키텍처 업그레이드 유효 |
| output36 | Qwen3-32B → -1.83점 퇴보 | 24GB GPU 한계: bf16 패치 강제 적용, lr/epoch 미최적화 |

---

## [output36] 2026-05-07~08 — Qwen3-32B QLoRA (최종 제출)

### 변경
- 모델: Qwen3-14B → Qwen3-32B
- config: config/config_qwen3_32b_traindev.yaml 추가

### 환경 이슈
- RTX 3090(24GB)에서 32B 4-bit 로드 시 prepare_model_for_kbit_training float32 캐스팅 OOM
- 해결: train_lora.py에 bf16 캐스팅 커스텀 함수로 대체

### 결과
- 제출 점수: **52.2273** (ROUGE-1: 0.6075, ROUGE-2: 0.4219, ROUGE-L: 0.5374)
- output35(54.0594) 대비 **-1.83점 퇴보**
- 최종 최고 점수: **output35 = 54.0594** (대회 종료)

### 교훈
- RTX 3090 24GB에서 32B QLoRA는 메모리 한계로 정상 학습 불가
- bf16 패치로 인한 layer norm 정밀도 저하 → 성능 하락 추정
- 14B가 이 GPU 환경에서의 실질적 상한선
