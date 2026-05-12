import os
import sys
import re
import argparse
import yaml

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data
from dataset import SummarizationDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output_name', type=str, default='output.csv',
                        help='저장할 파일명 (예: output2.csv)')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# BOS/EOS/PAD만 제거. #PersonN# 등 domain special token은 유지해야 함
# → 학습 summary의 85.9%가 #PersonN# 형식을 사용하므로, 평가 reference도 동일 형식
_REMOVE_TOKENS = ['<usr>', '<s>', '</s>', '<pad>']

def postprocess(text: str) -> str:
    for token in _REMOVE_TOKENS:
        text = text.replace(token, ' ')
    # 토크나이저가 '#Person2#는'을 '#Person2# 는'으로 디코딩하는 문제 수정
    # 다음 문자가 '#'가 아닐 때만 #...# 뒤 공백 제거
    text = re.sub(r'(#\w+#) (?!#)', r'\1', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def run_inference(config, checkpoint_path, output_name='output.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ic = config['inference']
    tc = config['tokenizer']

    print(f"체크포인트 로드: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    test_df = load_data(
        os.path.join(config['general']['data_path'], 'test.csv'),
        is_train=False,
    )

    test_dataset = SummarizationDataset(
        dialogues=test_df['dialogue'].tolist(),
        tokenizer=tokenizer,
        encoder_max_len=tc['encoder_max_len'],
        decoder_max_len=tc['decoder_max_len'],
        summaries=None,
    )

    fnames = test_df['fname'].tolist()
    all_summaries = []
    all_fnames = []
    batch_size = ic['batch_size']

    with torch.no_grad():
        for start in tqdm(range(0, len(test_dataset), batch_size), desc='Inference'):
            end = min(start + batch_size, len(test_dataset))
            batch_items = [test_dataset[i] for i in range(start, end)]
            batch_fnames = fnames[start:end]

            max_len = max(len(item['input_ids']) for item in batch_items)
            input_ids_list, attn_mask_list = [], []
            for item in batch_items:
                ids = item['input_ids']
                mask = item['attention_mask']
                pad_len = max_len - len(ids)
                input_ids_list.append(ids + [tokenizer.pad_token_id] * pad_len)
                attn_mask_list.append(mask + [0] * pad_len)

            input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
            attention_mask = torch.tensor(attn_mask_list, dtype=torch.long).to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                no_repeat_ngram_size=ic['no_repeat_ngram_size'],
                early_stopping=ic['early_stopping'],
                max_new_tokens=ic['generate_max_new_tokens'],
                num_beams=ic['num_beams'],
                length_penalty=ic['length_penalty'],
            )

            # skip_special_tokens=False: #PersonN# 등 domain 토큰 유지
            # postprocess에서 BOS/EOS/PAD만 수동 제거
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
            all_summaries.extend([postprocess(s) for s in decoded])
            all_fnames.extend(batch_fnames)

    result_path = config['general']['result_path']
    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, output_name)

    output_df = pd.DataFrame({'fname': all_fnames, 'summary': all_summaries})
    output_df.to_csv(output_path, index=False)

    # 길이 통계 출력
    lengths = output_df['summary'].str.len()
    print(f"\n추론 완료 → {output_path} ({len(output_df)}건)")
    print(f"예측 요약 길이: 평균 {lengths.mean():.1f}자 | 최대 {lengths.max()} | 최소 {lengths.min()}")
    return output_df


def main():
    args = parse_args()
    config = load_config(args.config)

    checkpoint_path = args.checkpoint or config['inference']['checkpoint_path']
    if not checkpoint_path:
        checkpoint_path = config['general']['output_dir']

    run_inference(config, checkpoint_path, output_name=args.output_name)


if __name__ == '__main__':
    main()
