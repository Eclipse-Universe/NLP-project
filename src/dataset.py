from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    """
    Seq2Seq 모델용 Dataset.
    DataCollatorForSeq2Seq와 함께 사용하며, padding은 collator가 배치 단위로 처리.
    (baseline처럼 전체 데이터를 한 번에 padding하면 메모리 낭비 + 느린 학습)

    train/val: dialogues + summaries 모두 전달
    test:      dialogues만 전달 (summaries=None)
    """

    def __init__(self, dialogues, tokenizer, encoder_max_len, decoder_max_len, summaries=None):
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

        # 인코더 입력 토크나이즈 (padding 없이 → collator가 처리)
        self.encoder_inputs = tokenizer(
            dialogues,
            max_length=encoder_max_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # 디코더 레이블 토크나이즈 (train/val에서만 사용)
        self.labels = None
        if summaries is not None:
            self.labels = tokenizer(
                summaries,
                max_length=decoder_max_len,
                truncation=True,
                padding=False,
                return_tensors=None,
            )

    def __len__(self):
        return len(self.encoder_inputs['input_ids'])

    def __getitem__(self, idx):
        item = {key: self.encoder_inputs[key][idx] for key in self.encoder_inputs}
        if self.labels is not None:
            # labels에 pad 토큰이 들어가면 DataCollatorForSeq2Seq가 -100으로 마스킹
            # → 모델이 pad 위치의 loss를 무시하게 됨
            item['labels'] = self.labels['input_ids'][idx]
        return item
