"""
Multi-model MBR (Minimum Bayes Risk) 앙상블.
각 샘플마다 후보 예측들 간 쌍방향 ROUGE를 계산하고,
평균 ROUGE가 가장 높은 예측을 선택합니다.
"""
import argparse
import pandas as pd
from rouge_score import rouge_scorer as rs


def mbr_score(scorer, hyp, candidates):
    refs = [c for c in candidates if c != hyp]
    if not refs:
        return 0.0
    scores = []
    for ref in refs:
        r = scorer.score(ref, hyp)
        scores.append((r['rouge1'].fmeasure + r['rouge2'].fmeasure + r['rougeL'].fmeasure) / 3)
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True, help='입력 CSV 경로 목록')
    parser.add_argument('--output', required=True, help='출력 CSV 경로')
    args = parser.parse_args()

    dfs = [pd.read_csv(p) for p in args.inputs]
    fnames = dfs[0]['fname'].tolist()
    for df in dfs[1:]:
        assert df['fname'].tolist() == fnames, "fname 순서 불일치"

    scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    summaries, sources = [], []
    n = len(fnames)
    for i in range(n):
        candidates = [str(df['summary'].iloc[i]) for df in dfs]
        scores = [mbr_score(scorer, c, candidates) for c in candidates]
        best = scores.index(max(scores))
        summaries.append(candidates[best])
        sources.append(best)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} 처리 완료")

    result = pd.DataFrame({'fname': fnames, 'summary': summaries})
    result.to_csv(args.output, index=False)

    lengths = result['summary'].str.len()
    print(f"\nMBR 앙상블 → {args.output} ({len(result)}건)")
    print(f"길이: avg {lengths.mean():.1f}자 | max {lengths.max()} | min {lengths.min()}")
    for i, path in enumerate(args.inputs):
        count = sources.count(i)
        print(f"  [{i}] {path.split('/')[-1]}: {count}건 ({count/len(sources)*100:.1f}%)")


if __name__ == '__main__':
    main()
