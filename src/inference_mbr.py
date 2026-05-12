"""
MBR (Minimum Bayes Risk) Decoding.
Generate N candidate summaries via temperature sampling, then select the
candidate with highest average ROUGE against all other candidates.
Reference: "Follow the Wisdom of the Crowd" (ACL Findings 2023)
"""
import os
import sys
import re
import argparse
import yaml

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge import Rouge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data


_REMOVE_TOKENS = ['<|im_end|>', '<|im_start|>', '<|endoftext|>',
                  '[|endofturn|]', '[|system|]', '[|user|]', '[|assistant|]']


def postprocess(text):
    for token in _REMOVE_TOKENS:
        text = text.replace(token, ' ')
    text = re.sub(r'(#\w+#) (?!#)', r'\1', text)
    return re.sub(r' {2,}', ' ', text).strip()


def mbr_select(candidates, rouge):
    valid = [c for c in candidates if c and len(c) > 1]
    if not valid:
        return candidates[0] if candidates else ""
    if len(valid) == 1:
        return valid[0]

    best_idx, best_score = 0, -1.0
    for i, c_i in enumerate(valid):
        others = [c for j, c in enumerate(valid) if j != i]
        try:
            scores = rouge.get_scores([c_i] * len(others), others, avg=True)
            avg = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
        except Exception:
            avg = 0.0
        if avg > best_score:
            best_score = avg
            best_idx = i
    return valid[best_idx]


def build_prompt(dialogue, system_prompt, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"다음 대화를 요약해주세요.\n\n{dialogue}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_qwen_v3.yaml')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output_name', type=str, default='output_mbr.csv')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ic = config['inference']
    system_prompt = config['system_prompt']
    max_input = config['tokenizer']['max_length'] - ic['max_new_tokens']

    print(f"MBR Decoding: N={args.n_samples}, temp={args.temperature}, top_p={args.top_p}")
    print(f"베이스 모델 로드: {config['general']['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(
        config['general']['model_name'], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    base_model = AutoModelForCausalLM.from_pretrained(
        config['general']['model_name'],
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )

    checkpoint_path = (args.checkpoint
                       or ic.get('checkpoint_path', '')
                       or config['general']['output_dir'])
    print(f"LoRA 어댑터 로드: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    test_df = load_data(
        os.path.join(config['general']['data_path'], 'test.csv'),
        is_train=False,
    )
    fnames = test_df['fname'].tolist()
    dialogues = test_df['dialogue'].tolist()

    rouge_scorer = Rouge()
    device = next(model.parameters()).device
    all_summaries = []
    n_valid_counts = []

    for dialogue in tqdm(dialogues, desc='MBR Inference'):
        prompt = build_prompt(dialogue, system_prompt, tokenizer)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_input)
        input_len = inputs['input_ids'].shape[1]

        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_new_tokens=ic['max_new_tokens'],
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.n_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

        candidates = []
        for i in range(args.n_samples):
            new_tokens = outputs[i][input_len:]
            summary = tokenizer.decode(new_tokens, skip_special_tokens=True)
            candidates.append(postprocess(summary))

        selected = mbr_select(candidates, rouge_scorer)
        all_summaries.append(selected)
        n_valid_counts.append(sum(1 for c in candidates if c and len(c) > 1))

    result_path = config['general']['result_path']
    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, args.output_name)

    output_df = pd.DataFrame({'fname': fnames, 'summary': all_summaries})
    output_df.to_csv(output_path, index=False)

    lengths = output_df['summary'].str.len()
    print(f"\nMBR 완료 → {output_path} ({len(output_df)}건)")
    print(f"길이: avg {lengths.mean():.1f}자 | max {lengths.max()} | min {lengths.min()}")
    print(f"후보 평균 유효수: {sum(n_valid_counts)/len(n_valid_counts):.1f}/{args.n_samples}")


if __name__ == '__main__':
    main()
