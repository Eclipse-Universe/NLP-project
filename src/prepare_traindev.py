"""train.csv + dev.csv → traindev.csv 통합 스크립트"""
import os
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/data')
    args = parser.parse_args()

    train_path = os.path.join(args.data_path, 'train.csv')
    dev_path = os.path.join(args.data_path, 'dev.csv')
    out_path = os.path.join(args.data_path, 'traindev.csv')

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    combined = pd.concat([train_df, dev_df], ignore_index=True)
    combined.to_csv(out_path, index=False)

    print(f"train: {len(train_df)}건 + dev: {len(dev_df)}건 = 통합: {len(combined)}건")
    print(f"저장: {out_path}")


if __name__ == '__main__':
    main()
