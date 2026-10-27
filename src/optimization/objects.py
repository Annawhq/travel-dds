from pathlib import Path
import numpy as np
import pandas as pd
from src.multicriteria.alternatives import PROJECT_ROOT

REQUIRED_COLUMNS = [
    "id", "name", "category", "lat", "lon", "visit_min", "price",
    "rating", "reviews", "open", "close", "indoor",
]

CATEGORIES = {
    "history", "religion", "architecture", "museum", "park",
    "walk", "food", "entertainment", "excursion",
}

# Грубые границы Казани и пригородов для проверки координат
LAT_RANGE = (55.5, 56.1)
LON_RANGE = (48.4, 49.5)

EARTH_RADIUS_KM = 6371.0

# Параметры расчета привлекательности
MIN_RELIABLE_REVIEWS = 500  # с какого количества отзывов рейтингу можно доверять
RATING_WEIGHT = 0.5         # вес качества (рейтинга); 1 − вес - популярность


def time_to_minutes(value: str) -> int:
    """Переводит строку 'ЧЧ:ММ' в минуты от полуночи"""
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def min_max(series: pd.Series) -> pd.Series:
    """Линейно переводит значения в диапазон [0, 1]"""
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(1.0, index=series.index)
    return (series - series.min()) / span


def compute_attractiveness(
    rating: pd.Series,
    reviews: pd.Series,
    min_reviews: int = MIN_RELIABLE_REVIEWS,
    rating_weight: float = RATING_WEIGHT,) -> pd.Series:
    """
    Привлекательность объекта по шкале [1, 5] из рейтинга и числа отзывов

    1. Качество - байесовский рейтинг: при малом числе отзывов рейтинг
       «подтягивается» к среднему по всем объектам, т.к. ему мало доверия
    2. Популярность - логарифм числа отзывов (разница между 100 и 1000
       отзывов важнее, чем между 50 000 и 51 000)
    3. Обе величины нормализуются в [0, 1], взвешиваются и
       переводятся на шкалу 1-5
    """
    mean_rating = (rating * reviews).sum() / reviews.sum()
    bayes_rating = (
        (reviews * rating + min_reviews * mean_rating)
        / (reviews + min_reviews)
    )
    popularity = np.log10(reviews + 1)
    score = (rating_weight * min_max(bayes_rating)
             + (1 - rating_weight) * min_max(popularity))
    return 1 + 4 * score


def load_objects(path: Path) -> pd.DataFrame:
    """
    Загружает туристические объекты и проверяет качество данных.
    Добавляет столбцы: open_min, close_min (минуты от полуночи)
    и attractiveness (привлекательность 1–5)
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В файле {path} нет столбцов: {missing}")
    df = df.dropna(how="all")
    gaps = df[REQUIRED_COLUMNS].isna()
    if gaps.any().any():
        problems = [
            f"  строка {row + 2} (id = {df.at[row, 'id']}): "
            + ", ".join(gaps.columns[gaps.loc[row]])
            for row in gaps.index[gaps.any(axis=1)]
        ]
        raise ValueError("В данных есть пропуски:\n" + "\n".join(problems))
    if df["id"].duplicated().any():
        raise ValueError("Идентификаторы объектов повторяются.")
    df["name"] = df["name"].str.strip()
    unknown = set(df["category"]) - CATEGORIES
    if unknown:
        raise ValueError(f"Неизвестные категории: {unknown}")

    if not df["lat"].between(*LAT_RANGE).all() \
            or not df["lon"].between(*LON_RANGE).all():
        raise ValueError("Координаты выходят за пределы Казани и пригородов.")

    if (df["visit_min"] <= 0).any():
        raise ValueError("Длительность посещения должна быть положительной.")
    if (df["price"] < 0).any():
        raise ValueError("Стоимость не может быть отрицательной.")
    if not df["indoor"].isin([0, 1]).all():
        raise ValueError("Признак indoor должен быть 0 или 1.")

    for column in ("rating", "reviews"):
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Столбец {column} должен быть числовым "
                "(дробная часть отделяется точкой: 4.8, а не 4,8)"
            )
    bad_rating = ~df["rating"].between(1, 5)
    if bad_rating.any():
        raise ValueError(
            f"Рейтинг должен быть в диапазоне [1, 5]: "
            f"{df.loc[bad_rating, 'id'].tolist()}"
        )
    bad_reviews = (df["reviews"] < 0) | (df["reviews"] % 1 != 0)
    if bad_reviews.any():
        raise ValueError(
            f"Число отзывов должно быть целым неотрицательным: "
            f"{df.loc[bad_reviews, 'id'].tolist()}"
        )

    df["open_min"] = df["open"].apply(time_to_minutes)
    close = df["close"].apply(time_to_minutes)
    # Закрытие в 00:00 или после полуночи (например, 10:00–02:00):
    # объект закрывается уже на следующие сутки, прибавляем 24 часа
    df["close_min"] = close.where(close > df["open_min"], close + 24 * 60)
    too_short = df["close_min"] - df["open_min"] < df["visit_min"]
    if too_short.any():
        details = df.loc[too_short, ["id", "open", "close", "visit_min"]]
        raise ValueError(
            "Объект не успеть посетить за время работы:\n"
            + details.to_string(index=False)
        )

    df["attractiveness"] = compute_attractiveness(df["rating"], df["reviews"])
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние по прямой между точками на сфере (км)"""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    objects = load_objects(PROJECT_ROOT / "data" / "raw" / "kazan_objects.csv")
    print(f"Загружено объектов: {len(objects)}")
    print()

    print("Объектов по категориям:")
    print(objects["category"].value_counts().to_string())
    print()

    kremlin = objects.set_index("id").loc["kremlin"]
    objects["km_from_kremlin"] = haversine_km(
        kremlin["lat"], kremlin["lon"], objects["lat"], objects["lon"]
    )
    print("Самые удаленные от Кремля объекты (км по прямой):")
    print(objects.nlargest(5, "km_from_kremlin")[["name", "km_from_kremlin"]]
          .round(1).to_string(index=False))
    print()

    columns = ["name", "rating", "reviews", "attractiveness"]
    print("Самые привлекательные объекты:")
    print(objects.nlargest(5, "attractiveness")[columns]
          .round(2).to_string(index=False))
    print("Наименее привлекательные объекты:")
    print(objects.nsmallest(5, "attractiveness")[columns]
          .round(2).to_string(index=False))
    print("Распределение привлекательности:")
    bins = pd.cut(objects["attractiveness"], [1, 2, 3, 4, 5], include_lowest=True)
    print(bins.value_counts().sort_index().to_string())
    print()

    print(f"Суммарное время посещения всех объектов: "
          f"{objects['visit_min'].sum() / 60:.1f} ч")
    print(f"Суммарная стоимость всех билетов: {objects['price'].sum()} руб.")
    print(f"Объектов в помещении: {objects['indoor'].sum()}, "
          f"на открытом воздухе: {(objects['indoor'] == 0).sum()}")
