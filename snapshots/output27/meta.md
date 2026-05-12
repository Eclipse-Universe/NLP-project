# output27 — output26 + output21 앙상블

## 점수
| 지표 | 값 |
|---|---|
| ROUGE-1 | 0.6139 |
| ROUGE-2 | 0.4362 |
| ROUGE-L | 0.5449 |
| **final_result** | **53.1651** |

## 설정
- 앙상블: output26(bias=0) + output21(bias=3)
- 선택: output26 192건(38.5%), output21 307건(61.5%)
- 평균 길이: 81.6자

## 결과 분석
- output21(53.4659) 대비 **-0.30점 하락**
- 원인: output26(52.42) < output21(53.47) → source 품질 열위로 앙상블 역효과

## 재현 명령
```bash
python src/ensemble.py \
  --inputs outputs/predictions/output26_v6ep2_nomnt.csv outputs/predictions/output21.csv \
  --priorities 0 3 \
  --output outputs/predictions/output27.csv
```
