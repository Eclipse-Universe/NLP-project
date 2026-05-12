import os
import sys
import re
import argparse
import yaml

import torch

# torch < 2.6 에는 nn.Module.set_submodule 미존재 — transformers 5.x 호환 패치
if not hasattr(torch.nn.Module, 'set_submodule'):
    def _set_submodule(self, target: str, module: torch.nn.Module):
        atoms = target.split('.')
        mod = self
        for atom in atoms[:-1]:
            mod = getattr(mod, atom)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_qwen.yaml')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output_name', type=str, default='output.csv')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_inference_prompt(dialogue, system_prompt, tokenizer, enable_thinking=False):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"다음 대화를 요약해주세요.\n\n{dialogue}"},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


_REMOVE_TOKENS = ['<|im_end|>', '<|im_start|>', '<|endoftext|>',
                  '[|endofturn|]', '[|system|]', '[|user|]', '[|assistant|]']

def postprocess(text: str) -> str:
    for token in _REMOVE_TOKENS:
        text = text.replace(token, ' ')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'(#\w+#) (?!#)', r'\1', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def run_inference(config, checkpoint_path, output_name='output.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ic = config['inference']
    system_prompt = config['system_prompt']
    max_input = config['tokenizer']['max_length'] - ic['max_new_tokens']

    print(f"베이스 모델 로드: {config['general']['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['general']['model_name'], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = config.get('use_4bit', False)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config['general']['model_name'],
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config['general']['model_name'],
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )

    print(f"LoRA 어댑터 로드: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    test_df = load_data(
        os.path.join(config['general']['data_path'], 'test.csv'),
        is_train=False,
    )

    fnames = test_df['fname'].tolist()
    dialogues = test_df['dialogue'].tolist()
    all_summaries = []

    model_name = config['general']['model_name']
    enable_thinking = 'Qwen3' in model_name or 'qwen3' in model_name.lower()

    for dialogue in tqdm(dialogues, desc='Inference'):
        prompt = build_inference_prompt(dialogue, system_prompt, tokenizer, enable_thinking=enable_thinking)
        inputs = tokenizer(
            prompt, return_tensors='pt', truncation=True, max_length=max_input
        )
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_new_tokens=ic['max_new_tokens'],
                min_new_tokens=ic.get('min_new_tokens', 0),
                do_sample=ic['do_sample'],
                num_beams=ic['num_beams'],
                length_penalty=ic.get('length_penalty', 1.0),
                no_repeat_ngram_size=ic.get('no_repeat_ngram_size', 0),
                pad_token_id=tokenizer.eos_token_id,
            )
        # 새로 생성된 토큰만 디코딩
        new_tokens = generated[0][inputs['input_ids'].shape[1]:]
        summary = tokenizer.decode(new_tokens, skip_special_tokens=True)
        all_summaries.append(postprocess(summary))

    result_path = config['general']['result_path']
    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, output_name)

    output_df = pd.DataFrame({'fname': fnames, 'summary': all_summaries})
    output_df.to_csv(output_path, index=False)

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
