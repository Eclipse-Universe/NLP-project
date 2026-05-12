# output16 — Qwen v3 + EXAONE v2 앙상블 (현재 최고)

## 점수
ROUGE-1: 0.6135 | ROUGE-2: 0.4387 | ROUGE-L: 0.5474 | **final_result: 53.3901**

## 설정
- 앙상블: output12(Qwen v3) 59.1% + output14(EXAONE v2) 40.9%
- 선택 기준: 샘플별 |len - 81.2자| 최소, output12 bias=3자 우선
- 예측 길이: avg 81.9자 (reference 81.2자에 가장 근접)

## 재현 명령
```bash
python3 - << 'EOF'
import pandas as pd
df12 = pd.read_csv('outputs/predictions/output12.csv')
df14 = pd.read_csv('outputs/predictions/output14.csv')
REF_AVG, BIAS = 81.2, 3
l12, l14 = df12['summary'].str.len(), df14['summary'].str.len()
use_14 = (abs(l14 - REF_AVG) < abs(l12 - REF_AVG) - BIAS)
summaries = [df14['summary'].iloc[i] if use_14.iloc[i] else df12['summary'].iloc[i] for i in range(len(df12))]
pd.DataFrame({'fname': df12['fname'], 'summary': summaries}).to_csv('outputs/predictions/output16.csv', index=False)
EOF
```

## git commit: 597eb86
