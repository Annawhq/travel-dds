import numpy as np
import pandas as pd
from src.forecast.explore import MONTH_NAMES, monthly_series
from src.forecast.forecast import RECENT_FROM_YEAR, fit_temperature, predict_temperature
from src.forecast.model import to_table
from src.forecast.weather import WEATHER_PATH, load_weather, read_metadata
from src.multicriteria.alternatives import PROJECT_ROOT

NEW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "kazan_weather_2026.csv"
CHECK_YEAR = 2026


def only_full_months(weather: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только месяцы, для которых есть данные за все дни"""
    days = weather.groupby(weather.index.to_period("M")).size()
    full = [p for p, n in days.items() if n == p.days_in_month]
    return weather[weather.index.to_period("M").isin(full)]


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    old_meta, new_meta = read_metadata(WEATHER_PATH), read_metadata(NEW_DATA_PATH)
    if (old_meta["latitude"], old_meta["longitude"]) != \
            (new_meta["latitude"], new_meta["longitude"]):
        raise ValueError("Новые данные выгружены для другой точки сетки.")

    history = to_table(monthly_series(load_weather()))
    new = only_full_months(load_weather(NEW_DATA_PATH))
    actual = to_table(monthly_series(new))
    actual = actual[actual["year"] == CHECK_YEAR]
    print(f"Новые данные: {new.index.min().date()} — {new.index.max().date()}, "
          f"полных месяцев {CHECK_YEAR} года: {len(actual)}")
    print()

    # Модель обучена ТОЛЬКО на 1991–2025, как и прогноз на дашборде
    model = fit_temperature(history)
    forecast = predict_temperature(model, actual["year"], actual["month"])
    climate = history.groupby("month")["t_mean"].mean()
    rain_norm = (history[history["year"] >= RECENT_FROM_YEAR]
                 .groupby("month")["rainy_share"].mean())

    result = pd.DataFrame({
        "Месяц": actual["month"].map(lambda m: MONTH_NAMES[m - 1]).to_numpy(),
        "Факт, °C": actual["t_mean"].to_numpy(),
        "Прогноз, °C": forecast["t_forecast"].to_numpy(),
        "Ошибка, °C": actual["t_mean"].to_numpy() - forecast["t_forecast"].to_numpy(),
        "В интервале": (actual["t_mean"].to_numpy() >= forecast["t_low"].to_numpy())
                       & (actual["t_mean"].to_numpy() <= forecast["t_high"].to_numpy()),
        "Норма, °C": actual["month"].map(climate).to_numpy(),
        "Дожд. дни: факт": actual["rainy_share"].to_numpy(),
        "Дожд. дни: прогноз": actual["month"].map(rain_norm).to_numpy(),
    })
    print(f"Проверка прогноза на {CHECK_YEAR} год (данные, которых модель не видела):")
    print(result.round(2).to_string(index=False))
    print()

    model_error = result["Ошибка, °C"]
    norm_error = result["Факт, °C"] - result["Норма, °C"]
    rain_error = result["Дожд. дни: факт"] - result["Дожд. дни: прогноз"]
    print(f"Модель «норма + тренд»: MAE = {np.abs(model_error).mean():.2f} °C, "
          f"RMSE = {np.sqrt((model_error ** 2).mean()):.2f} °C, "
          f"смещение = {model_error.mean():+.2f} °C")
    print(f"Климатическая норма:    MAE = {np.abs(norm_error).mean():.2f} °C, "
          f"RMSE = {np.sqrt((norm_error ** 2).mean()):.2f} °C, "
          f"смещение = {norm_error.mean():+.2f} °C")
    print(f"Попадание в 80%-й интервал: {result['В интервале'].sum()} "
          f"из {len(result)} ({result['В интервале'].mean():.0%})")
    print(f"Доля дождливых дней: MAE = {np.abs(rain_error).mean():.3f}")
