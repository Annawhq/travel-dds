from pathlib import Path
import pandas as pd
from src.multicriteria.alternatives import PROJECT_ROOT

WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "kazan_weather_daily.csv"

# Служебные строки выгрузки Open-Meteo перед таблицей:
# заголовок метаданных, строка метаданных, пустая строка
METADATA_LINES = 3

COLUMNS = {
    "time": "date",
    "temperature_2m_mean (°C)": "t_mean",
    "temperature_2m_max (°C)": "t_max",
    "temperature_2m_min (°C)": "t_min",
    "precipitation_sum (mm)": "precip_mm",
    "precipitation_hours (h)": "precip_hours",
}

# Физически правдоподобные диапазоны для Казани
VALID_RANGES = {
    "t_mean": (-50, 45),
    "t_max": (-50, 45),
    "t_min": (-55, 40),
    "precip_mm": (0, 200),
    "precip_hours": (0, 24),
}

RAINY_DAY_MM = 1.0   # день с осадками - сумма осадков ≥ 1 мм


def read_metadata(path: Path) -> dict:
    """Координаты и высота точки, для которой выгружены данные"""
    header = pd.read_csv(path, nrows=1)
    return header.iloc[0].to_dict()


def load_weather(path: Path = WEATHER_PATH) -> pd.DataFrame:
    """
    Загружает суточные данные о погоде и проверяет их качество:
    состав столбцов, пропуски, повторы и пропущенные даты,
    физическую правдоподобность значений, согласованность t_min ≤ t_mean ≤ t_max
    """
    df = pd.read_csv(path, skiprows=METADATA_LINES)

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"В файле нет столбцов: {missing}")
    df = df.rename(columns=COLUMNS)[list(COLUMNS.values())]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    if df["date"].duplicated().any():
        raise ValueError("В данных есть повторяющиеся даты.")
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    absent = full_range.difference(df["date"])
    if len(absent):
        raise ValueError(f"Пропущено дат: {len(absent)}, первая: {absent[0].date()}")

    gaps = df.isna().sum()
    if gaps.any():
        raise ValueError(f"Пропуски значений:\n{gaps[gaps > 0]}")

    for column, (low, high) in VALID_RANGES.items():
        bad = ~df[column].between(low, high)
        if bad.any():
            raise ValueError(
                f"{column}: {bad.sum()} значений вне диапазона [{low}, {high}]"
            )

    inconsistent = ~((df["t_min"] <= df["t_mean"]) & (df["t_mean"] <= df["t_max"]))
    if inconsistent.any():
        raise ValueError(f"Нарушено t_min ≤ t_mean ≤ t_max в {inconsistent.sum()} днях")

    df["rainy"] = df["precip_mm"] >= RAINY_DAY_MM
    return df.set_index("date")


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    meta = read_metadata(WEATHER_PATH)
    print(f"Точка данных: {meta['latitude']}, {meta['longitude']}, "
          f"высота {meta['elevation']} м, часовой пояс {meta['timezone']}")

    weather = load_weather()
    print(f"Период: {weather.index.min().date()} — {weather.index.max().date()}, "
          f"дней: {len(weather)}")
    days_per_year = weather.groupby(weather.index.year).size()
    print(f"Лет: {len(days_per_year)}, дней в году: от {days_per_year.min()} "
          f"до {days_per_year.max()}")
    print("Проверки качества пройдены: пропусков, повторов и "
          "неправдоподобных значений нет.")
    print()

    print("Описательная статистика:")
    print(weather.drop(columns="rainy").describe().round(1).to_string())
    print()

    climate = weather.groupby(weather.index.month).agg(
        t_mean=("t_mean", "mean"),
        precip_mm_per_month=("precip_mm", lambda s: s.sum() / weather.index.year.nunique()),
        rainy_share=("rainy", "mean"),
    )
    climate.index.name = "месяц"
    print("Климатическая норма 1991–2025 по месяцам:")
    print(climate.round(2).to_string())
