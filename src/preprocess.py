import re
import pandas as pd


def clean_dialogue(text: str) -> str:
    """
    데이터에 존재하는 두 가지 노이즈를 정규화.
    - '\\n' (백슬래시-n 리터럴) → 실제 줄바꿈 \n
    - '<br>' HTML 태그 → 실제 줄바꿈 \n
    """
    text = text.replace('\\n', '\n')
    text = re.sub(r'<br\s*/?>', '\n', text)
    return text.strip()


def load_data(file_path: str, is_train: bool = True) -> pd.DataFrame:
    """
    CSV 파일을 읽고 dialogue 노이즈를 정제하여 반환.
    is_train=True  → fname, dialogue, summary 컬럼 반환
    is_train=False → fname, dialogue 컬럼 반환
    """
    df = pd.read_csv(file_path)
    df['dialogue'] = df['dialogue'].apply(clean_dialogue)

    if is_train:
        return df[['fname', 'dialogue', 'summary']]
    else:
        return df[['fname', 'dialogue']]
