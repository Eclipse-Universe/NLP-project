import os
import sys
import argparse
import yaml
import re
import numpy as np

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
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training as _prepare_model_for_kbit_training

def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    # bf16 대신 float32 캐스팅 → 24GB GPU에서 32B 모델 OOM 방지
    for name, param in model.named_parameters():
        if param.ndim == 1:
            param.data = param.data.to(torch.bfloat16)
    if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {})
        model.enable_input_require_grads()
    return model
from rouge import Rouge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_qwen.yaml')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_prompt(dialogue, system_prompt, tokenizer, summary=None, enable_thinking=False):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"다음 대화를 요약해주세요.\n\n{dialogue}"},
    ]
    if summary is not None:
        messages.append({"role": "assistant", "content": summary})
    kwargs = dict(tokenize=False, add_generation_prompt=(summary is None))
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def make_dataset(df, tokenizer, config, is_train):
    system_prompt = config['system_prompt']
    max_length = config['tokenizer']['max_length']
    model_name = config['general']['model_name']
    enable_thinking = 'Qwen3' in model_name or 'qwen3' in model_name.lower()
    input_ids_list, attention_mask_list, labels_list = [], [], []

    for _, row in df.iterrows():
        full_text = build_prompt(
            row['dialogue'], system_prompt, tokenizer,
            summary=row['summary'] if is_train else None,
            enable_thinking=enable_thinking,
        )
        full = tokenizer(full_text, max_length=max_length, truncation=True, padding=False)

        if is_train:
            # prompt 부분만 찾아서 -100 마스킹
            prompt_only = build_prompt(row['dialogue'], system_prompt, tokenizer, enable_thinking=enable_thinking)
            prompt = tokenizer(prompt_only, max_length=max_length, truncation=True, padding=False)
            prompt_len = len(prompt['input_ids'])
            labels = [-100] * prompt_len + full['input_ids'][prompt_len:]
        else:
            labels = full['input_ids'][:]

        input_ids_list.append(full['input_ids'])
        attention_mask_list.append(full['attention_mask'])
        labels_list.append(labels[:max_length])

    return Dataset.from_dict({
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list,
    })


class CausalLMCollator:
    """right-padding 기반 causal LM collator."""
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(len(f['input_ids']) for f in features)
        max_len = min(max_len, self.max_length)

        batch_input_ids, batch_attn, batch_labels = [], [], []
        for f in features:
            ids = f['input_ids'][:max_len]
            mask = f['attention_mask'][:max_len]
            labs = f['labels'][:max_len]
            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [self.pad_id] * pad_len)
            batch_attn.append(mask + [0] * pad_len)
            batch_labels.append(labs + [-100] * pad_len)

        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attn, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
        }


_REMOVE = ['<|im_end|>', '<|im_start|>', '<|endoftext|>',
           '[|endofturn|]', '[|system|]', '[|user|]', '[|assistant|]']

def postprocess(text):
    for t in _REMOVE:
        text = text.replace(t, ' ')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'(#\w+#) (?!#)', r'\1', text)
    return re.sub(r' {2,}', ' ', text).strip()


class RougeEvalCallback(TrainerCallback):
    """epoch 종료 시 beam search 기반 ROUGE를 계산하고, best ROUGE 모델을 직접 저장.
    추론 조건(beam=4, lp=0.7)과 동일하게 평가해 checkpoint 선택 불일치 방지."""
    def __init__(self, model, tokenizer, val_df, config, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        eval_n = config.get('rouge_eval_samples', len(val_df))
        self.val_df = val_df.sample(min(len(val_df), eval_n), random_state=42)
        self.system_prompt = config['system_prompt']
        ic = config['inference']
        self.max_new_tokens = ic['max_new_tokens']
        self.num_beams = ic.get('num_beams', 1)
        self.length_penalty = ic.get('length_penalty', 1.0)
        self.min_new_tokens = ic.get('min_new_tokens', 0)
        self.max_input = config['tokenizer']['max_length'] - self.max_new_tokens
        self.rouge = Rouge()
        self.best_rouge_avg = 0.0
        self.best_rouge_dir = os.path.join(output_dir, 'best_rouge_checkpoint')
        model_name = config['general']['model_name']
        self.enable_thinking = 'Qwen3' in model_name or 'qwen3' in model_name.lower()

    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        self.model.eval()
        preds, refs = [], []

        for _, row in self.val_df.iterrows():
            prompt = build_prompt(row['dialogue'], self.system_prompt, self.tokenizer, enable_thinking=self.enable_thinking)
            inputs = self.tokenizer(
                prompt, return_tensors='pt', truncation=True, max_length=self.max_input
            )
            device = next(self.model.parameters()).device
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=inputs['input_ids'].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs['input_ids'].shape[1]:]
            summary = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            preds.append(postprocess(summary))
            refs.append(row['summary'])

        valid = [(p, r) for p, r in zip(preds, refs) if p and r]
        if valid:
            scores = self.rouge.get_scores([p for p,r in valid], [r for p,r in valid], avg=True)
            r1 = scores['rouge-1']['f']
            r2 = scores['rouge-2']['f']
            rl = scores['rouge-l']['f']
            rouge_avg = r1 + r2 + rl
            metrics = {
                'eval_rouge-1': r1,
                'eval_rouge-2': r2,
                'eval_rouge-l': rl,
                'eval_rouge_avg': rouge_avg,
            }
            print(f"\n[Epoch {state.epoch:.0f}] ROUGE-1: {r1:.4f} | ROUGE-2: {r2:.4f} | ROUGE-L: {rl:.4f} | avg: {rouge_avg:.4f}")
            print(f"[PRED] {preds[0][:100]}")
            print(f"[GOLD] {refs[0][:100]}")

            if rouge_avg > self.best_rouge_avg:
                self.best_rouge_avg = rouge_avg
                os.makedirs(self.best_rouge_dir, exist_ok=True)
                self.model.save_pretrained(self.best_rouge_dir)
                self.tokenizer.save_pretrained(self.best_rouge_dir)
                print(f"  → Best ROUGE 갱신 ({rouge_avg:.4f}), 저장: {self.best_rouge_dir}")
            print()
            wandb.log(metrics, step=state.global_step)

        torch.cuda.empty_cache()
        self.model.train()


def main():
    args = parse_args()
    config = load_config(args.config)

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
    )

    print(f"모델: {config['general']['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(
        config['general']['model_name'], trust_remote_code=True, padding_side='right'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_file = config['general'].get('train_file', 'train.csv')
    val_file = config['general'].get('val_file', 'dev.csv')
    train_df = load_data(os.path.join(config['general']['data_path'], train_file))
    val_df = load_data(os.path.join(config['general']['data_path'], val_file))
    print(f"Train: {len(train_df)}건 | Val: {len(val_df)}건")

    print("데이터셋 토크나이징...")
    train_dataset = make_dataset(train_df, tokenizer, config, is_train=True)
    val_dataset = make_dataset(val_df, tokenizer, config, is_train=True)
    print(f"완료: train {len(train_dataset)}건 | val {len(val_dataset)}건")

    print("모델 로드 중...")
    use_4bit = config.get('use_4bit', False)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config['general']['model_name'],
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config['training']['gradient_checkpointing']
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config['general']['model_name'],
            dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )

    lc = config['lora']
    lora_config = LoraConfig(
        r=lc['r'],
        lora_alpha=lc['lora_alpha'],
        lora_dropout=lc['lora_dropout'],
        bias=lc['bias'],
        target_modules=lc['target_modules'],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if not use_4bit and config['training']['gradient_checkpointing']:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    tc = config['training']
    output_dir = config['general']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tc['num_train_epochs'],
        learning_rate=tc['learning_rate'],
        per_device_train_batch_size=tc['per_device_train_batch_size'],
        per_device_eval_batch_size=tc['per_device_eval_batch_size'],
        gradient_accumulation_steps=tc['gradient_accumulation_steps'],
        warmup_ratio=tc['warmup_ratio'],
        weight_decay=tc['weight_decay'],
        lr_scheduler_type=tc['lr_scheduler_type'],
        optim=tc['optim'],
        eval_strategy=tc['evaluation_strategy'],
        save_strategy=tc['save_strategy'],
        save_total_limit=tc['save_total_limit'],
        bf16=tc['bf16'],
        load_best_model_at_end=tc['load_best_model_at_end'],
        metric_for_best_model='eval_loss',  # loss 기반 best 선택 (ROUGE는 callback으로 추적)
        greater_is_better=False,
        seed=tc['seed'],
        logging_dir=tc.get('logging_dir', '/root/outputs/logs_qwen'),
        logging_strategy=tc['logging_strategy'],
        logging_steps=tc['logging_steps'],
        report_to='wandb',
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    data_collator = CausalLMCollator(tokenizer, config['tokenizer']['max_length'])

    rouge_callback = RougeEvalCallback(model, tokenizer, val_df, config, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[rouge_callback],
    )

    print("학습 시작...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"모델 저장 완료 → {output_dir}")

    wandb.finish()


if __name__ == '__main__':
    main()
