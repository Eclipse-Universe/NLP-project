# output23 — output21 + output22 앙상블

## 점수
| 지표 | 값 |
|---|---|
| ROUGE-1 | 0.6106 |
| ROUGE-2 | 0.4380 |
| ROUGE-L | 0.5451 |
| **final_result** | **53.1235** |

## 설정
- 앙상블 구성: output21 (bias=0) + output22 (bias=3)
- 선택 비율: output21 482건(96.4%), output22 17건(3.6%)
- 평균 길이: 82.0자

## 앙상블 명령
```bash
python src/ensemble.py \
  --inputs outputs/predictions/output21.csv outputs/predictions/output22.csv \
  --priorities 0 3 \
  --output outputs/predictions/output23.csv
```

## 결과 분석
- output21(53.4659) 대비 **-0.34점 퇴보**
- 원인: output22(52.7315) < output21(53.4659) → 낮은 품질 source 혼합이 오염 효과

## 교훈
- **앙상블 황금률**: source 단독 성능이 현재 최고보다 낮으면 앙상블해도 하락
- 길이 기반 앙상블은 content-agnostic → 더 나쁜 예측이 길이만 맞으면 선택될 수 있음

## git hash
b533a91 (output22/23 생성 + v6 OOM 수정 재시작)
