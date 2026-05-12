# output29 — MBR(output28) + output21 앙상블

## 점수
제출 예정 — 제출 후 업데이트

## 설정
- 앙상블: output28 MBR(bias=0, 우선) + output21(bias=3)
- 선택: output28 205건(41.1%), output21 294건(58.9%)
- 평균 길이: **81.2자** (reference 평균과 정확히 일치, 역대 최근접)

## 앙상블 명령
```bash
python src/ensemble.py \
  --inputs outputs/predictions/output28_mbr_n10.csv outputs/predictions/output21.csv \
  --priorities 0 3 \
  --output outputs/predictions/output29.csv
```

## 기대 효과
- MBR이 beam=4보다 ref에 더 가까운 길이(82.1자 vs 83.8자) 생성
- 앙상블 후 avg 81.2자로 ref와 정확히 일치
- MBR 합의 선택이 내용 품질도 개선되면 output21(53.47) 초과 기대
- 목표: 53.5+

## 점수
| 지표 | 값 |
|---|---|
| ROUGE-1 | 0.6019 |
| ROUGE-2 | 0.4243 |
| ROUGE-L | 0.5348 |
| **final_result** | **52.0300** |

output21(53.4659) 대비 **-1.44점** — output28 오염으로 큰 폭 하락
