import os
import sys
import argparse
import yaml

import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data
from dataset import SummarizationDataset
from evaluate import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_tokenizer_and_model(config):
    model_name = config['general']['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # baseline은 Person1~3만 등록 → 데이터에 최대 Person7까지 존재하므로 전부 등록
    # special token으로 등록하면 tokenizer가 이 문자열을 분해하지 않고 단일 토큰으로 처리
    special_tokens = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens)

    bart_config = BartConfig.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)

    # special token 추가 후 embedding 테이블 크기를 맞춰줘야 함
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def build_datasets(config, tokenizer):
    data_path = config['general']['data_path']
    enc_max = config['tokenizer']['encoder_max_len']
    dec_max = config['tokenizer']['decoder_max_len']

    train_df = load_data(os.path.join(data_path, 'train.csv'), is_train=True)
    val_df   = load_data(os.path.join(data_path, 'dev.csv'),   is_train=True)

    train_dataset = SummarizationDataset(
        dialogues=train_df['dialogue'].tolist(),
        summaries=train_df['summary'].tolist(),
        tokenizer=tokenizer,
        encoder_max_len=enc_max,
        decoder_max_len=dec_max,
    )
    val_dataset = SummarizationDataset(
        dialogues=val_df['dialogue'].tolist(),
        summaries=val_df['summary'].tolist(),
        tokenizer=tokenizer,
        encoder_max_len=enc_max,
        decoder_max_len=dec_max,
    )

    print(f"Train: {len(train_dataset)}건 | Val: {len(val_dataset)}건")
    return train_dataset, val_dataset


def build_trainer(config, model, tokenizer, train_dataset, val_dataset):
    tc = config['training']

    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        num_train_epochs=tc['num_train_epochs'],
        learning_rate=tc['learning_rate'],
        per_device_train_batch_size=tc['per_device_train_batch_size'],
        per_device_eval_batch_size=tc['per_device_eval_batch_size'],
        gradient_accumulation_steps=tc['gradient_accumulation_steps'],
        warmup_ratio=tc['warmup_ratio'],
        weight_decay=tc['weight_decay'],
        lr_scheduler_type=tc['lr_scheduler_type'],
        optim=tc['optim'],
        evaluation_strategy=tc['evaluation_strategy'],
        save_strategy=tc['save_strategy'],
        save_total_limit=tc['save_total_limit'],
        fp16=tc['fp16'],
        load_best_model_at_end=tc['load_best_model_at_end'],
        metric_for_best_model=tc['metric_for_best_model'],
        greater_is_better=tc['greater_is_better'],
        seed=tc['seed'],
        logging_dir=tc['logging_dir'],
        logging_strategy=tc['logging_strategy'],
        predict_with_generate=tc['predict_with_generate'],
        generation_max_length=tc['generation_max_length'],
        do_train=True,
        do_eval=True,
        report_to='none',   # wandb는 나중에 추가
    )

    # DataCollatorForSeq2Seq:
    # - 배치 내 시퀀스들을 가장 긴 것 기준으로 padding
    # - labels의 pad 위치를 -100으로 마스킹 (loss 계산에서 제외)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=tc['early_stopping_patience'],
                early_stopping_threshold=tc['early_stopping_threshold'],
            )
        ],
    )

    return trainer


def main():
    args = parse_args()
    config = load_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | Model: {config['general']['model_name']}")

    tokenizer, model = build_tokenizer_and_model(config)
    train_dataset, val_dataset = build_datasets(config, tokenizer)
    trainer = build_trainer(config, model, tokenizer, train_dataset, val_dataset)

    print("학습 시작...")
    trainer.train()

    # best model을 output_dir에 저장
    trainer.save_model(config['general']['output_dir'])
    tokenizer.save_pretrained(config['general']['output_dir'])
    print(f"Best model 저장 완료 → {config['general']['output_dir']}")


if __name__ == '__main__':
    main()
