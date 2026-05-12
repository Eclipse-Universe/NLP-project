# Dialogue Summarization 진행 기록

---

## v1.1 — Inference 버그 수정 (2026-04-24)

### 제출 결과 (output1.csv)
- ROUGE-1: 0.2510 | ROUGE-2: 0.1134 | ROUGE-L: 0.1838 | **final_result: 18.27**

### 원인 분석으로 발견한 3가지 버그

**버그 1 (치명적): skip_special_tokens=True로 #PersonN# 토큰 제거**
- 학습 summary의 85.9% (10,697/12,457)가 `#Person1#`, `#Person2#` 형식 사용
- `skip_special_tokens=True`로 디코딩 시 이 토큰들이 공백으로 제거됨
- 평가 reference의 `#Person1#은` vs 예측의 ` 은` → ROUGE에서 완전 불일치
- **fix**: `skip_special_tokens=False` + BOS/EOS/PAD 수동 제거

**버그 2 (심각): 토크나이저 디코딩 공백 문제**
- 모델이 `#Person1#은`을 생성하면 토크나이저가 `#Person1# 은`으로 디코딩 (공백 삽입)
- reference의 `#Person1#은`(1토큰)과 예측의 `#Person1#`+`은`(2토큰) → ROUGE 불일치
- **fix**: `re.sub(r'(#\w+#) (?!#)', r'\1', text)` 후처리 추가

**버그 3 (주요): 예측 길이가 정답의 3배**
- 예측 평균: 235.9자 / 정답 평균: 81.2자 (토큰 기준: 92 vs 31)
- `digit82/kobart-summarization`이 뉴스 요약 스타일(장문)로 사전학습됨
- **fix**: `max_new_tokens=30, length_penalty=0.5` (grid search로 최적값 탐색)

### 출력 파일 비교
| 파일 | 핵심 변경 | 평균길이 | dev avg×100 |
|---|---|---|---|
| output1.csv | 원본 (버그 전부 포함) | 235.9자 | 9.29 (실제 18.27) |
| output2.csv | lp=0.9, n=3 (미미한 개선) | 207.1자 | - |
| output3.csv | max=60, lp=0.8 | 119.2자 | - |
| output4.csv | max=45, lp=0.7 | 87.0자 | - |
| output5.csv | skip=False + BOS제거 | 286.1자 | - |
| **output6.csv** | **버그 1,2,3 전부 수정** | **78.6자** | **17.85→예상 35+** |

### 실행 방법
```bash
python src/inference.py --config config/config.yaml --output_name output6.csv
```

### 다음 단계 (v2.0 예정)
- [ ] output6 제출 후 점수 확인
- [ ] 재학습: decoder max_len을 학습 시에도 30~35로 맞춤
- [ ] 더 나은 모델 실험 (pko-t5-large, EXAONE+LoRA)

---

## v1.0 — KoBART Fine-tuning 코드 구축 (2026-04-24)

### 목표
Baseline의 6가지 문제점을 해결하고, 재사용 가능한 스크립트 기반 구조로 재설계.

### 변경사항 (vs Baseline)
| 항목 | Baseline | v1.0 |
|---|---|---|
| 코드 구조 | Jupyter Notebook (단일 파일) | 모듈 분리 스크립트 |
| Special tokens | Person1~3만 등록 | Person1~7 + 기존 토큰 전부 등록 |
| encoder_max_len | 512 | 1024 (긴 대화 truncation 방지) |
| 노이즈 전처리 | 없음 | `\\n`→`\n`, `<br>`→`\n` 처리 |
| Dataset padding | 전체 데이터 일괄 padding | DataCollatorForSeq2Seq (배치 단위) |
| 평가 지표 | 단일 reference ROUGE | rouge_avg(R1+R2+RL) 합산, 빈 예측 필터링 |
| best model 기준 | validation loss | rouge_avg (높을수록 좋음) |
| wandb | 설정 필요 | 비활성화 (추후 추가) |

### 파일 구조
```
/root/
├── config/config.yaml       # 모든 하이퍼파라미터
├── src/
│   ├── preprocess.py        # 노이즈 제거, 데이터 로딩
│   ├── dataset.py           # SummarizationDataset
│   ├── evaluate.py          # ROUGE 계산 (단일/다중 reference)
│   ├── train.py             # 학습 메인
│   └── inference.py         # 추론 및 제출 파일 생성
├── outputs/
│   ├── checkpoints/         # 모델 체크포인트
│   ├── predictions/         # output.csv (제출용)
│   └── logs/                # 학습 로그
└── progress.md              # 이 파일
```

### 실행 방법
```bash
# 학습
cd /root
python src/train.py --config config/config.yaml

# 추론 (학습 완료 후)
python src/inference.py --config config/config.yaml
# 또는 특정 체크포인트 지정
python src/inference.py --config config/config.yaml --checkpoint outputs/checkpoints/checkpoint-XXXX
```

### 주요 하이퍼파라미터
- 모델: `digit82/kobart-summarization`
- encoder_max_len: 1024
- batch_size: 8 (gradient_accumulation=4 → 유효 배치 32)
- learning_rate: 1e-5 (cosine decay)
- epochs: 10 (EarlyStopping patience=3)

### 학습 결과 (epoch별 dev ROUGE)
| Epoch | Train Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | rouge_avg |
|---|---|---|---|---|---|
| 1 | 2.692 | 0.2018 | 0.0468 | 0.1961 | 0.4447 |
| 2 | 1.879 | 0.2127 | 0.0537 | 0.2074 | 0.4738 |
| 3 | 1.728 | 0.2239 | 0.0598 | 0.2192 | 0.5029 |
| 4 | 1.615 | 0.2230 | 0.0595 | 0.2192 | 0.5018 |
| 5 | 1.541 | 0.2260 | 0.0610 | 0.2217 | 0.5087 |
| **6** | **1.476** | **0.2280** | **0.0636** | **0.2224** | **0.5141 ← Best** |
| 7 | 1.435 | 0.2269 | 0.0624 | 0.2213 | 0.5106 |
| 8 | 1.402 | 0.2277 | 0.0638 | 0.2210 | 0.5125 |
| 9 | 1.389 | 0.2275 | 0.0639 | 0.2214 | 0.5128 |

- EarlyStopping: epoch 9에서 종료 (patience=3, best=epoch6)
- 총 학습 시간: 약 886초 (약 15분)
- Best checkpoint: `/root/outputs/checkpoints/` (epoch6 모델)
- 추론 결과: `/root/outputs/predictions/output.csv` (499건)

### 관찰된 문제점 (v2.0에서 수정 필요)
- 일부 예측 요약이 generate_max_length=100으로 인해 중간에 잘림
- 일부 예측이 지나치게 장황함 (no_repeat_ngram_size=2가 충분하지 않음)
- #PersonN# 토큰이 skip_special_tokens=True로 제거되면서 이름 자리가 공백이 되는 경우 존재

### 다음 단계 (v2.0 예정)
- [ ] generate_max_length 상향 (100 → 150)
- [ ] 장황한 요약 문제 분석 및 개선
- [ ] paust/pko-t5-large 또는 EXAONE 실험
- [ ] wandb 연동

---

<!-- 새 버전은 위에 추가 (최신이 맨 위) -->
