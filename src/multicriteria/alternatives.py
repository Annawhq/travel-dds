from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRITERIA_COLUMNS = [
    "cost",
    "weather",
    "transport",
    "attractiveness",
    "interests",
]


def load_alternatives(path: Path) -> tuple[list[str], np.ndarray]:
    """
    Загружает альтернативы и их оценки по критериям
    (шкала 1-5, 5 - наиболее предпочтительное значение)
    """
    df = pd.read_csv(path)

    missing = [c for c in ["city", *CRITERIA_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"В файле {path} нет столбцов: {missing}")
    if df["city"].duplicated().any():
        raise ValueError("В файле есть повторяющиеся города.")

    scores = df[CRITERIA_COLUMNS]
    if scores.isna().any().any():
        raise ValueError("В оценках есть пропуски.")
    if not scores.isin([1, 2, 3, 4, 5]).all().all():
        raise ValueError("Оценки должны быть целыми числами от 1 до 5.")

    return df["city"].tolist(), scores.to_numpy(dtype=float)
