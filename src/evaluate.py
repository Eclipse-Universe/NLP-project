import numpy as np
from rouge import Rouge


rouge = Rouge()


def compute_metrics(pred, tokenizer):
    """
    Seq2SeqTrainer의 compute_metrics 콜백 함수.

    [Insight] 대회 평가는 3개 reference 평균이지만, dev.csv에는 summary가 1개뿐.
    따라서 학습 중 val ROUGE는 단일 reference 기준이며, 실제 제출 점수보다 낮게 나올 수 있음.
    경향(오르는지 내리는지)을 보는 용도로 사용.

    metric_for_best_model = 'rouge_avg' (rouge-1 + rouge-2 + rouge-l 합산)
    """
    predictions = pred.predictions
    labels = pred.label_ids

    # -100은 loss 계산 시 무시되는 pad 위치 → pad_token_id로 복원 후 디코딩
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # rouge 라이브러리는 빈 문자열 입력 시 오류 → 필터링
    valid = [(p, l) for p, l in zip(decoded_preds, decoded_labels) if p and l]
    if not valid:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'rouge_avg': 0.0}

    preds, golds = zip(*valid)
    results = rouge.get_scores(list(preds), list(golds), avg=True)

    r1 = results['rouge-1']['f']
    r2 = results['rouge-2']['f']
    rl = results['rouge-l']['f']

    # 샘플 출력 (학습 중 실제 예측 품질을 눈으로 확인)
    print(f"\n[PRED] {decoded_preds[0]}")
    print(f"[GOLD] {decoded_labels[0]}")
    print(f"ROUGE-1: {r1:.4f} | ROUGE-2: {r2:.4f} | ROUGE-L: {rl:.4f}\n")

    return {
        'rouge-1': r1,
        'rouge-2': r2,
        'rouge-l': rl,
        'rouge_avg': r1 + r2 + rl,  # best model 선택 기준
    }


def evaluate_multi_ref(predictions, references_list):
    """
    다중 reference ROUGE 계산 (제출 전 dev 성능 최종 확인용).

    references_list: [[ref1_1, ref1_2, ...], [ref2_1, ref2_2, ...], ...]
    각 예측에 대해 여러 reference와 비교한 평균을 구함.

    [Insight] 대회 채점 방식과 동일한 구조.
    실제 test 제출 후 점수와 가장 가까운 추정치를 낼 수 있음.
    """
    total_r1, total_r2, total_rl = 0.0, 0.0, 0.0
    count = 0

    for pred, refs in zip(predictions, references_list):
        if not pred.strip():
            continue
        valid_refs = [r for r in refs if r.strip()]
        if not valid_refs:
            continue

        r1s, r2s, rls = [], [], []
        for ref in valid_refs:
            try:
                score = rouge.get_scores(pred, ref)[0]
                r1s.append(score['rouge-1']['f'])
                r2s.append(score['rouge-2']['f'])
                rls.append(score['rouge-l']['f'])
            except Exception:
                continue

        if r1s:
            total_r1 += np.mean(r1s)
            total_r2 += np.mean(r2s)
            total_rl += np.mean(rls)
            count += 1

    if count == 0:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'rouge_avg': 0.0}

    r1 = total_r1 / count
    r2 = total_r2 / count
    rl = total_rl / count
    return {'rouge-1': r1, 'rouge-2': r2, 'rouge-l': rl, 'rouge_avg': r1 + r2 + rl}
