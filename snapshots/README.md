# Snapshots — output별 재현 정보

각 output의 config, 체크포인트, 점수, git commit hash를 보관합니다.
코드 자체는 git history에, 예측 CSV는 outputs/predictions/에 있습니다.
각 output의 상세 정보는 해당 폴더의 meta.md를 참조하세요.

---

## 점수 요약 (내림차순)

| output | 모델/설정 | 점수 | git commit |
|---|---|---|---|
| **output30** | **Qwen2.5-14B 4-bit QLoRA Epoch2** | **53.5031 ← 현재 최고** | 5de7e33 |
| output21 | output16+output20 앙상블 | 53.4659 (전 최고) | 616ca1f |
| output16 | Qwen v3 + EXAONE v2 앙상블 | 53.3901 | 597eb86 |
| output19 | 3-way 앙상블 (v3+EXAONE+v4) | 53.3206 | b0fcb1f |
| **output12** | **Qwen v3 r=32 epoch2** | **53.3054 ← 7B 단독 최고** | f453a08 |
| output14 | EXAONE v2 lp=0.7 | 53.1547 | aabc7e8 |
| output32 | output30+output21 역방향 앙상블 | 53.2959 — 단독보다 퇴보 | 885a02f |
| output31 | output21+output30 앙상블 | 53.1858 — bias 오류로 퇴보 | 5de7e33 |
| output23 | output21+output22 앙상블 | 53.1235 | b533a91 |
| output15 | EXAONE v2 lp=0.65 | 52.9385 | 81ef322 |
| output11 | Qwen v2 r=32 epoch2 | 52.8828 | — |
| output18 | Qwen v4+v3 앙상블 | 52.8486 | b0fcb1f |
| output22 | Qwen v3 + min_new_tokens=25 | 52.7315 | b533a91 |
| output20 | Qwen v5 (train+dev) epoch2 | 52.5391 | 5d7b37d |
| output13 | EXAONE v1 (weight tying 파괴) | 52.5744 | 9fa0e07 |
| output07 | Qwen v1 r=16 lp=0.7 | 52.58 | — |
| output26 | Qwen v6 ep2 진단 | 52.4188 | — |
| output27 | output26+21 앙상블 | 53.1651 | — |
| output17 | Qwen v4 r=64 epoch2 | 51.7074 | b0fcb1f |
| output25 | output24+21 앙상블 | 52.1864 | — |
| output24 | Qwen v6 beam 콜백 + mnt25 (역대 최저) | 51.0192 | — |
| output29 | MBR+output21 앙상블 | 52.0300 — output28 오염 | 33f582a |
| output28 | MBR Decoding N=10 (temp=0.8) | 50.9351 — 온도 샘플링 역효과 | 33f582a |

---

## 디렉토리 구조

```
snapshots/
  output12/  ← Qwen v3 7B 단독 최고 (53.30)
  output13/  ← EXAONE v1 실패 (weight tying 파괴)
  output14/  ← EXAONE v2 수정 후
  output15/  ← EXAONE v2 lp=0.65
  output16/  ← Qwen v3 + EXAONE 앙상블
  output17/  ← Qwen v4 r=64 실패 (style drift)
  output18/  ← v4+v3 앙상블 (퇴보)
  output19/  ← 3-way 앙상블
  output20/  ← Qwen v5 (train+dev)
  output21/  ← 7B 앙상블 최고 (53.47)
  output22/  ← Qwen v3 + min_new_tokens=25 실패 (52.73)
  output23/  ← output21+22 앙상블 퇴보 (53.12)
  output24/  ← Qwen v6 beam 콜백 + mnt25 (51.02, 역대 최저)
  output25/  ← output24+21 앙상블 (52.19, 퇴보)
  output26/  ← v6 ep2 진단 (52.42, v3 ep2보다 0.88점 열위)
  output27/  ← output26+21 앙상블 (53.17, 퇴보)
  output28/  ← MBR N=10 (50.94, 온도 샘플링 역효과)
  output29/  ← output28+21 앙상블 (52.03, 역효과)
  output30/  ← Qwen2.5-14B 4-bit QLoRA Epoch2 (53.50, 현재 최고)
  output31/  ← output21+output30 앙상블 (53.19, bias 오류로 퇴보)
  output32/  ← output30+output21 역방향 앙상블 (제출 대기)
```
