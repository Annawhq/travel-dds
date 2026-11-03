import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.forecast.explore import FIGURES_DIR, monthly_series
from src.forecast.weather import load_weather

TEST_YEARS = range(2016, 2026)   # годы проверки (скользящий прогноз на год вперед)


def to_table(monthly: pd.DataFrame) -> pd.DataFrame:
    """Месячный ряд → таблица с колонками year, month и показателями"""
    table = monthly.copy()
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.reset_index(drop=True)


def climatology(train, test, column):
    """Климатическая норма: среднее значение месяца за все годы обучения"""
    norm = train.groupby("month")[column].mean()
    return test["month"].map(norm).to_numpy()


def seasonal_naive(train, test, column):
    """Сезонный наивный прогноз: как в этом месяце последнего известного года"""
    last = train[train["year"] == train["year"].max()].set_index("month")[column]
    return test["month"].map(last).to_numpy()


def monthly_trend(train, test, column):
    """Норма + свой линейный тренд для каждого месяца (12 отдельных регрессий)"""
    forecast = []
    for _, row in test.iterrows():
        same_month = train[train["month"] == row["month"]]
        slope, intercept = np.polyfit(same_month["year"], same_month[column], 1)
        forecast.append(intercept + slope * row["year"])
    return np.array(forecast)


def common_trend(train, test, column):
    """
    Норма месяца + общий для всех месяцев линейный тренд,
    оцененный по отклонениям от нормы (одна регрессия по всем данным)
    """
    norm = train.groupby("month")[column].mean()
    anomaly = train[column] - train["month"].map(norm)
    slope, intercept = np.polyfit(train["year"], anomaly, 1)
    return (test["month"].map(norm) + intercept + slope * test["year"]).to_numpy()


MODELS = {
    "Климатическая норма": climatology,
    "Сезонный наивный": seasonal_naive,
    "Норма + тренд каждого месяца": monthly_trend,
    "Норма + общий тренд": common_trend,
}


def backtest(table, column, model, test_years=TEST_YEARS):
    """
    Скользящая проверка: для каждого года Y модель обучается на годах < Y
    и прогнозирует все 12 месяцев года Y. Так имитируется реальная
    ситуация: турист планирует поездку на год вперед
    """
    parts = []
    for year in test_years:
        train = table[table["year"] < year]
        test = table[table["year"] == year]
        result = test[["year", "month", column]].copy()
        result["forecast"] = model(train, test, column)
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def metrics(result, column) -> dict:
    error = result[column] - result["forecast"]
    return {
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "Смещение": error.mean(),
    }


def compare_models(table, column) -> pd.DataFrame:
    rows = {name: metrics(backtest(table, column, model), column)
            for name, model in MODELS.items()}
    return pd.DataFrame(rows).T


def plot_backtest(table, path) -> None:
    baseline = backtest(table, "t_mean", climatology)
    best = backtest(table, "t_mean", common_trend)
    dates = pd.to_datetime(dict(year=best["year"], month=best["month"], day=1))
    plt.figure(figsize=(11, 4.5))
    plt.plot(dates, best["t_mean"], label="Факт", color="black", linewidth=1.2)
    plt.plot(dates, baseline["forecast"], label="Климатическая норма",
             linestyle=":", color="gray")
    plt.plot(dates, best["forecast"], label="Норма + общий тренд",
             linestyle="--", color="#c0392b")
    plt.ylabel("Средняя температура месяца, °C")
    plt.title("Проверка прогноза на год вперед, 2016–2025")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    table = to_table(monthly_series(load_weather()))

    print(f"Скользящая проверка: годы {TEST_YEARS.start}–{TEST_YEARS.stop - 1}, "
          f"прогноз на 12 месяцев вперед, {len(TEST_YEARS) * 12} прогнозов")
    print()
    print("Средняя температура месяца, °C:")
    print(compare_models(table, "t_mean").round(3).to_string())
    print()
    print("Доля дней с осадками ≥ 1 мм:")
    print(compare_models(table, "rainy_share").round(3).to_string())

    plot_backtest(table, FIGURES_DIR / "weather_backtest.png")
    print()
    print(f"График сохранен в {FIGURES_DIR / 'weather_backtest.png'}")
