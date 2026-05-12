"""
앙상블: 여러 output CSV에서 샘플별로 최적 예측을 선택합니다.
선택 기준: reference 평균 길이(81.2자)에 가장 가까운 예측 선택.
우선순위 편향(bias): 품질이 높은 모델을 우선함.

사용 예시 (현재 최고: output21 재현):
    python src/ensemble.py \
        --inputs outputs/predictions/output16.csv outputs/predictions/output20.csv \
        --priorities 0 3 \
        --output outputs/predictions/output21_repro.csv

사용 예시 (2-way 기본):
    python src/ensemble.py \
        --inputs outputs/predictions/output12.csv outputs/predictions/output14.csv \
        --output outputs/predictions/my_ensemble.csv
"""
import argparse
import pandas as pd


REF_AVG_LEN = 81.2  # 학습 데이터 reference 요약 평균 길이


def parse_args():
    parser = argparse.ArgumentParser(description='길이 기반 앙상블')
    parser.add_argument(
        '--inputs', nargs='+', required=True,
        help='입력 CSV 경로 목록 (fname, summary 컬럼 필요). 앞에 있을수록 우선순위 높음.'
    )
    parser.add_argument(
        '--priorities', nargs='+', type=float, default=None,
        help='각 파일의 우선순위 편향 (작을수록 우선). 미지정 시 순서대로 0,1,2,...'
    )
    parser.add_argument(
        '--ref_avg', type=float, default=REF_AVG_LEN,
        help=f'reference 평균 길이 (기본: {REF_AVG_LEN}자)'
    )
    parser.add_argument(
        '--output', type=str, default='outputs/predictions/ensemble.csv',
        help='출력 CSV 경로'
    )
    return parser.parse_args()


def load_predictions(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        assert 'fname' in df.columns and 'summary' in df.columns, \
            f"{p}: fname, summary 컬럼이 필요합니다."
        dfs.append(df)

    fnames_ref = dfs[0]['fname'].tolist()
    for i, df in enumerate(dfs[1:], 1):
        assert df['fname'].tolist() == fnames_ref, \
            f"파일 {i+1}의 fname 순서가 첫 번째 파일과 다릅니다."
    return dfs


def ensemble(dfs, priorities, ref_avg):
    n = len(dfs[0])
    selected_summaries = []
    selected_sources = []

    for i in range(n):
        best_summary = None
        best_score = float('inf')
        best_src = -1

        for src_idx, (df, bias) in enumerate(zip(dfs, priorities)):
            summary = df['summary'].iloc[i]
            length_dist = abs(len(str(summary)) - ref_avg) + bias
            if length_dist < best_score:
                best_score = length_dist
                best_summary = summary
                best_src = src_idx

        selected_summaries.append(best_summary)
        selected_sources.append(best_src)

    return selected_summaries, selected_sources


def main():
    args = parse_args()

    if args.priorities is None:
        priorities = list(range(len(args.inputs)))
    else:
        assert len(args.priorities) == len(args.inputs), \
            "--priorities 개수가 --inputs 개수와 일치해야 합니다."
        priorities = args.priorities

    print(f"입력 파일: {len(args.inputs)}개")
    for path, bias in zip(args.inputs, priorities):
        print(f"  bias={bias:+.1f}  {path}")
    print(f"reference 평균 길이: {args.ref_avg}자\n")

    dfs = load_predictions(args.inputs)
    summaries, sources = ensemble(dfs, priorities, args.ref_avg)

    result = pd.DataFrame({'fname': dfs[0]['fname'], 'summary': summaries})
    result.to_csv(args.output, index=False)

    lengths = result['summary'].str.len()
    print(f"앙상블 결과 → {args.output} ({len(result)}건)")
    print(f"길이: avg {lengths.mean():.1f}자 | max {lengths.max()} | min {lengths.min()}")
    for i, (path, bias) in enumerate(zip(args.inputs, priorities)):
        count = sources.count(i)
        print(f"  [{i}] bias={bias:+.1f} {path}: {count}건 ({count/len(sources)*100:.1f}%)")


if __name__ == '__main__':
    main()
