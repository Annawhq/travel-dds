from pathlib import Path
import numpy as np
import pandas as pd
from src.multicriteria.alternatives import PROJECT_ROOT

REQUIRED_COLUMNS = [
    "id", "name", "category", "lat", "lon", "visit_min", "price",
    "attractiveness", "open", "close", "indoor",
]

CATEGORIES = {
    "history", "religion", "architecture", "museum", "park",
    "walk", "food", "entertainment", "excursion",
}

# Грубые границы Казани и пригородов для проверки координат
LAT_RANGE = (55.5, 56.1)
LON_RANGE = (48.4, 49.5)

EARTH_RADIUS_KM = 6371.0


def time_to_minutes(value: str) -> int:
    """Переводит строку 'ЧЧ:ММ' в минуты от полуночи"""
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def load_objects(path: Path) -> pd.DataFrame:
    """
    Загружает туристические объекты и проверяет качество данных.
    Время работы переводится в минуты от полуночи
    (столбцы open_min, close_min)
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

    unknown = set(df["category"]) - CATEGORIES
    if unknown:
        raise ValueError(f"Неизвестные категории: {unknown}")

    if not df["lat"].between(*LAT_RANGE).all() \
            or not df["lon"].between(*LON_RANGE).all():
        raise ValueError("Координаты выходят за пределы Казани и пригородов")

    if (df["visit_min"] <= 0).any():
        raise ValueError("Длительность посещения должна быть положительной")
    if (df["price"] < 0).any():
        raise ValueError("Стоимость не может быть отрицательной.")
    if not pd.api.types.is_numeric_dtype(df["attractiveness"]):
        raise ValueError(
            "Привлекательность должна быть числом "
            "(проверьте, что дробная часть отделена точкой: 4.8, а не 4,8)."
        )
    out_of_range = ~df["attractiveness"].between(1, 5)
    if out_of_range.any():
        raise ValueError(
            f"Привлекательность должна быть в диапазоне [1, 5]: "
            f"{df.loc[out_of_range, 'id'].tolist()}"
        )
    if not df["indoor"].isin([0, 1]).all():
        raise ValueError("Признак indoor должен быть 0 или 1")

    df["open_min"] = df["open"].apply(time_to_minutes)
    close = df["close"].apply(time_to_minutes)
    df["close_min"] = close.where(close > df["open_min"], close + 24 * 60)
    too_short = df["close_min"] - df["open_min"] < df["visit_min"]
    if too_short.any():
        raise ValueError(
            f"Объект не успеть посетить за время работы: "
            f"{df.loc[too_short, 'id'].tolist()}"
        )

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

    print(f"Суммарное время посещения всех объектов: "
          f"{objects['visit_min'].sum() / 60:.1f} ч")
    print(f"Суммарная стоимость всех билетов: {objects['price'].sum()} руб.")
    print(f"Объектов в помещении: {objects['indoor'].sum()}, "
          f"на открытом воздухе: {(objects['indoor'] == 0).sum()}")
