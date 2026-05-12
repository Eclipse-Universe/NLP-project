#!/bin/bash
# output34: Qwen2.5-14B QLoRA — train+dev 통합 학습
# output30 동일 하이퍼파라미터, 데이터만 확장 (train+dev → epoch2 직접)
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_NAME="output34"
CONFIG="config/config_qwen14b_traindev.yaml"
CHECKPOINT_DIR="/root/outputs/checkpoints_qwen14b_traindev/best_rouge_checkpoint"
RESULT_DIR="/root/outputs/predictions"
SNAPSHOT_DIR="$REPO_DIR/snapshots/$OUTPUT_NAME"
GIT_TOKEN=""

echo "========================================"
echo " output34 학습 파이프라인 시작"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 디스크 체크 (150GB 초과 시 경고)
DISK_USED=$(df / | awk 'NR==2 {print $3}')
DISK_USED_GB=$((DISK_USED / 1024 / 1024))
echo "[디스크] 현재 사용: ${DISK_USED_GB}GB"
if [ "$DISK_USED_GB" -gt 150 ]; then
    echo "⚠️  경고: 디스크 사용량 ${DISK_USED_GB}GB > 150GB"
    echo "    계속 진행하려면 Enter, 중단하려면 Ctrl+C"
    read -r
fi

cd "$REPO_DIR"

# Step 1: traindev.csv 생성
echo ""
echo "[Step 1] train+dev 통합 데이터 생성"
python src/prepare_traindev.py --data_path /root/data

# Step 2: 학습
echo ""
echo "[Step 2] 학습 시작 (epoch=2, beam=4, lp=0.7)"
echo "[디스크] $(df -h / | awk 'NR==2 {print $3"/"$2" ("$5" used)"}')"
python src/train_lora.py --config "$CONFIG"

# Step 3: 추론
echo ""
echo "[Step 3] 추론 시작"
echo "[디스크] $(df -h / | awk 'NR==2 {print $3"/"$2" ("$5" used)"}')"
mkdir -p "$RESULT_DIR"
python src/inference_lora.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT_DIR" \
    --output_name "${OUTPUT_NAME}.csv"

# Step 4: CSV 확인
CSV_PATH="$RESULT_DIR/${OUTPUT_NAME}.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo "❌ CSV 생성 실패: $CSV_PATH"
    exit 1
fi
echo ""
echo "[Step 4] CSV 생성 완료: $CSV_PATH"
python3 -c "
import pandas as pd
df = pd.read_csv('$CSV_PATH')
lengths = df['summary'].str.len()
print(f'  행: {len(df)}건 | 평균길이: {lengths.mean():.1f}자 | min: {lengths.min()} | max: {lengths.max()}')
print(df.head(3).to_string())
"

# Step 5: meta.md 생성
echo ""
echo "[Step 5] 메타 파일 생성"
mkdir -p "$SNAPSHOT_DIR"
cp "$CSV_PATH" "$SNAPSHOT_DIR/${OUTPUT_NAME}.csv"
cp "$CONFIG" "$SNAPSHOT_DIR/config_used.yaml"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
AVG_LEN=$(python3 -c "import pandas as pd; df=pd.read_csv('$CSV_PATH'); print(f'{df[\"summary\"].str.len().mean():.1f}')")

cat > "$SNAPSHOT_DIR/meta.md" << METAEOF
# ${OUTPUT_NAME} — Qwen2.5-14B QLoRA (train+dev 통합, Epoch 2)

## 점수
미제출 (제출 대기)

## 설정
- 모델: Qwen/Qwen2.5-14B-Instruct
- 방법: 4-bit NF4 QLoRA (output30과 동일 하이퍼파라미터)
- LoRA: r=32, alpha=64, dropout=0.05, target_modules 7개
- 학습 데이터: train.csv + dev.csv 통합 (traindev.csv)
- 학습: lr=2e-4, epoch=2 (sweet spot 직접 지정), batch=1×GA16=16
- max_length: 768, rouge_eval_samples: 100
- 추론: beam=4, length_penalty=0.7, max_new_tokens=100

## output30과의 차이
- 데이터: train only → **train+dev 통합 (+약 500샘플)**
- epoch: 3 (best_rouge 선택) → **2 (직접 지정)**
- 나머지 모두 동일

## 예측 요약 길이
- 평균: ${AVG_LEN}자 (ref avg: 81.2자)

## 재현
\`\`\`bash
python src/prepare_traindev.py --data_path /root/data
python src/train_lora.py --config config/config_qwen14b_traindev.yaml
python src/inference_lora.py \\
  --config config/config_qwen14b_traindev.yaml \\
  --checkpoint /root/outputs/checkpoints_qwen14b_traindev/best_rouge_checkpoint \\
  --output_name output34.csv
\`\`\`

## 생성일
${TIMESTAMP}
METAEOF

echo "  meta.md 저장: $SNAPSHOT_DIR/meta.md"

# Step 6: progress 로그 생성
echo ""
echo "[Step 6] 진행 로그 생성"
PROGRESS_FILE="$REPO_DIR/progress/$(date '+%y-%m-%d').md"
cat > "$PROGRESS_FILE" << LOGEOF
# $(date '+%Y-%m-%d') 작업 기록

## output34 — Qwen2.5-14B train+dev 통합 재학습

### 목적
- output30(53.5031) 대비 데이터 증가로 성능 향상 시도
- MBR/앙상블이 반복 실패 → 단독 강한 모델 데이터 확장 전략

### 설정
- output30과 동일 하이퍼파라미터 (r=32, lr=2e-4, beam=4, lp=0.7)
- 학습 데이터: train+dev 통합 (traindev.csv)
- epoch=2 직접 지정 (sweet spot 검증됨)

### 파일
- config: config/config_qwen14b_traindev.yaml
- 결과: snapshots/output34/output34.csv

### 결과
- 제출 점수: (제출 후 업데이트)
LOGEOF

echo "  progress 저장: $PROGRESS_FILE"

# Step 7: git push
echo ""
echo "[Step 7] Git push"
cd "$REPO_DIR"
git config --global user.email "dopago2000@gmail.com"
git config --global user.name "Eclipse-Universe"
git remote set-url origin "https://Eclipse-Universe:${GIT_TOKEN}@github.com/Eclipse-Universe/dialogue-summarization-nlp-4.git"

git add \
    "config/config_qwen14b_traindev.yaml" \
    "src/prepare_traindev.py" \
    "snapshots/${OUTPUT_NAME}/" \
    "progress/$(date '+%y-%m-%d').md" \
    "notebooks/" \
    "snapshots/external_results/" \
    "snapshots/output33/output33.csv" \
    "configs_external/" \
    "run_output34.sh" 2>/dev/null || true

git add "notebooks/" "snapshots/external_results/" "configs_external/" 2>/dev/null || true

git status --short
git commit -m "output34: Qwen2.5-14B QLoRA train+dev 통합 (epoch=2, beam=4, lp=0.7)

- output30 동일 하이퍼파라미터, train+dev 데이터 통합
- 노트북, 외부 결과 CSV, adapter config 정리 추가
- prepare_traindev.py, config_qwen14b_traindev.yaml 신규"

git push origin main
echo ""
echo "========================================"
echo " ✅ output34 완료 & git push 성공"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " CSV: $CSV_PATH"
echo "========================================"
echo ""
echo "[최종 디스크] $(df -h / | awk 'NR==2 {print $3"/"$2" ("$5" used)"}')"
