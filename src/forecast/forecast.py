import numpy as np
import pandas as pd
from src.forecast.explore import MONTH_NAMES, monthly_series
from src.forecast.model import TEST_YEARS, to_table
from src.forecast.weather import load_weather
from src.multicriteria.alternatives import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FORECAST_YEARS = [2026, 2027]
INTERVAL_Z = 1.2816        # 80%-й интервал при нормальном распределении ошибок
TRIP_DAYS = 2              # длительность поездки (как в профилях этапов 4-5)
RECENT_FROM_YEAR = 2011    # для осадков берем последние 15 лет (есть тренд)


def fit_temperature(train: pd.DataFrame) -> dict:
    """
    Модель «норма месяца + общий тренд» и разброс остатков по месяцам.
    Разброс нужен для интервального прогноза
    """
    norm = train.groupby("month")["t_mean"].mean()
    anomaly = train["t_mean"] - train["month"].map(norm)
    slope, intercept = np.polyfit(train["year"], anomaly, 1)
    residual = anomaly - (intercept + slope * train["year"])
    spread = residual.groupby(train["month"]).std()
    return {"norm": norm, "slope": slope, "intercept": intercept, "spread": spread}


def predict_temperature(model: dict, years, months) -> pd.DataFrame:
    """Точечный прогноз и 80%-й интервал для пар (год, месяц)"""
    years, months = np.asarray(years), np.asarray(months)
    point = (pd.Series(months).map(model["norm"]).to_numpy()
             + model["intercept"] + model["slope"] * years)
    half_width = INTERVAL_Z * pd.Series(months).map(model["spread"]).to_numpy()
    return pd.DataFrame({
        "year": years, "month": months,
        "t_forecast": point,
        "t_low": point - half_width,
        "t_high": point + half_width,
    })


def interval_coverage(table: pd.DataFrame) -> float:
    """
    Проверка интервала на скользящих прогнозах 2016–2025:
    какая доля фактических значений попала в 80%-й интервал
    """
    hits = []
    for year in TEST_YEARS:
        train = table[table["year"] < year]
        test = table[table["year"] == year]
        forecast = predict_temperature(fit_temperature(train),
                                       test["year"], test["month"])
        inside = ((test["t_mean"].to_numpy() >= forecast["t_low"].to_numpy())
                  & (test["t_mean"].to_numpy() <= forecast["t_high"].to_numpy()))
        hits.extend(inside)
    return float(np.mean(hits))


def rain_scenarios(weather: pd.DataFrame, trip_days: int = TRIP_DAYS,
                   from_year: int = RECENT_FROM_YEAR) -> pd.DataFrame:
    """
    Вероятности погодных сценариев для поездки длиной trip_days дней
    по историческим данным: перебираем все окна из trip_days подряд идущих
    дней внутри месяца и считаем, сколько в окне дней с осадками ≥ 1 мм.
      «лучше»   — ни одного дождливого дня,
      «обычная» — дождь в части дней,
      «хуже»    — дождь во все дни поездки
    """
    recent = weather[weather.index.year >= from_year]
    rows = []
    for month in range(1, 13):
        counts = []
        for _, days in recent[recent.index.month == month].groupby(
                recent[recent.index.month == month].index.year):
            rainy = days["rainy"].astype(int)
            counts.extend(rainy.rolling(trip_days).sum().dropna())
        counts = np.array(counts)
        rows.append({
            "month": month,
            "p_better": np.mean(counts == 0),
            "p_normal": np.mean((counts > 0) & (counts < trip_days)),
            "p_worse": np.mean(counts == trip_days),
            "windows": len(counts),
        })
    return pd.DataFrame(rows)


def dashboard_table(table, model, rain_norm) -> pd.DataFrame:
    """Факт 1991–2025 и прогноз на FORECAST_YEARS в одной таблице для BI"""
    fact = table[["year", "month", "t_mean", "precip_mm", "rainy_share"]].copy()
    fact["kind"] = "факт"
    future = predict_temperature(
        model,
        np.repeat(FORECAST_YEARS, 12),
        np.tile(range(1, 13), len(FORECAST_YEARS)),
    ).rename(columns={"t_forecast": "t_mean"})
    future["rainy_share"] = future["month"].map(rain_norm)
    future["kind"] = "прогноз"
    result = pd.concat([fact, future], ignore_index=True)
    result.insert(0, "date", pd.to_datetime(
        dict(year=result["year"], month=result["month"], day=1)))
    result["month_name"] = result["month"].map(lambda m: MONTH_NAMES[m - 1])
    numeric = result.select_dtypes("number").columns
    result[numeric] = result[numeric].round(2)
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    weather = load_weather()
    table = to_table(monthly_series(weather))

    model = fit_temperature(table)
    print(f"Модель обучена на 1991–2025: тренд {model['slope'] * 10:+.2f} °C за 10 лет")
    print(f"Покрытие 80%-го интервала на проверке 2016–2025: "
          f"{interval_coverage(table):.0%}")
    print()

    rain_norm = (table[table["year"] >= RECENT_FROM_YEAR]
                 .groupby("month")["rainy_share"].mean())
    forecast = predict_temperature(
        model, [FORECAST_YEARS[0]] * 12, range(1, 13))
    forecast["rainy_share"] = forecast["month"].map(rain_norm)
    forecast["month"] = forecast["month"].map(lambda m: MONTH_NAMES[m - 1])
    print(f"Прогноз на {FORECAST_YEARS[0]} год (температура, °C; "
          f"доля дождливых дней — норма {RECENT_FROM_YEAR}–2025):")
    print(forecast.drop(columns="year").round(2).to_string(index=False))
    print()

    scenarios = rain_scenarios(weather)
    print(f"Вероятности погодных сценариев для поездки на {TRIP_DAYS} дня "
          f"(по данным {RECENT_FROM_YEAR}–2025):")
    shown = scenarios.copy()
    shown["month"] = shown["month"].map(lambda m: MONTH_NAMES[m - 1])
    print(shown.round(3).to_string(index=False))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_table(table, model, rain_norm).to_csv(
        PROCESSED_DIR / "weather_dashboard.csv", index=False, encoding="utf-8")
    scenarios.round(3).to_csv(
        PROCESSED_DIR / "weather_scenarios.csv", index=False, encoding="utf-8")
    print()
    print(f"Таблицы для дашборда сохранены в {PROCESSED_DIR}")
