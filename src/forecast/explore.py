import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from src.forecast.weather import load_weather
from src.multicriteria.alternatives import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MONTH_NAMES = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
               "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]


def monthly_series(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегация суточных данных по месяцам:
    средняя температура, сумма осадков, доля дней с осадками ≥ 1 мм
    """
    monthly = weather.resample("MS").agg({
        "t_mean": "mean",
        "precip_mm": "sum",
        "rainy": "mean",
    })
    monthly.columns = ["t_mean", "precip_mm", "rainy_share"]
    return monthly


def trend(series: pd.Series) -> str:
    """Линейный тренд по годам: наклон за 10 лет, p-значение, R²"""
    result = linregress(series.index, series.values)
    p_value = "< 0.001" if result.pvalue < 0.001 else f"= {result.pvalue:.3f}"
    return (f"{result.slope * 10:+.3f} за 10 лет, p {p_value}, "
            f"R² = {result.rvalue ** 2:.2f}")


def plot_annual_trend(annual: pd.Series, path) -> None:
    result = linregress(annual.index, annual.values)
    plt.figure(figsize=(9, 4.5))
    plt.plot(annual.index, annual.values, marker="o", label="Среднегодовая температура")
    plt.plot(annual.index, result.intercept + result.slope * annual.index,
             linestyle="--",
             label=f"Линейный тренд: {result.slope * 10:+.2f} °C за 10 лет")
    plt.xlabel("Год")
    plt.ylabel("°C")
    plt.title("Казань: среднегодовая температура, 1991–2025")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_seasonal_profile(monthly: pd.DataFrame, path) -> None:
    groups = [monthly.loc[monthly.index.month == m, "t_mean"] for m in range(1, 13)]
    plt.figure(figsize=(9, 4.5))
    plt.boxplot(groups)
    plt.xticks(range(1, 13), MONTH_NAMES)
    plt.ylabel("Средняя температура месяца, °C")
    plt.title("Казань: сезонный профиль температуры (разброс по 35 годам)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    weather = load_weather()
    monthly = monthly_series(weather)
    print(f"Месячный ряд: {len(monthly)} месяцев "
          f"({monthly.index.min():%Y-%m} — {monthly.index.max():%Y-%m})")
    print()

    # 1. Сезонность: насколько меняется норма в течение года
    climate = monthly.groupby(monthly.index.month)["t_mean"].mean()
    print(f"Сезонность: норма от {climate.min():.1f} °C до {climate.max():.1f} °C, "
          f"размах {climate.max() - climate.min():.1f} °C")

    # 2. Межгодовая изменчивость: разброс одного и того же месяца по годам
    spread = monthly.groupby(monthly.index.month)["t_mean"].std()
    print(f"Межгодовой разброс (СКО месяца по годам): "
          f"от {spread.min():.2f} °C до {spread.max():.2f} °C")
    print()

    # 3. Тренд
    annual = weather.groupby(weather.index.year)["t_mean"].mean()
    print("Тренд среднегодовой температуры:", trend(annual))
    for month in (6, 7, 8):
        series = monthly.loc[monthly.index.month == month, "t_mean"]
        series.index = series.index.year
        print(f"Тренд температуры, {MONTH_NAMES[month - 1]}:", trend(series))
    summer = monthly[monthly.index.month.isin([6, 7, 8])]
    summer_rain = summer.groupby(summer.index.year)["rainy_share"].mean()
    print("Тренд доли дождливых дней летом:", trend(summer_rain))
    print()

    # 4. Остаточная автокорреляция: есть ли «память» в отклонениях от нормы
    anomaly = monthly["t_mean"] - monthly.groupby(monthly.index.month)["t_mean"].transform("mean")
    print(f"Автокорреляция отклонений от нормы: "
          f"лаг 1 мес = {anomaly.autocorr(1):.2f}, лаг 12 мес = {anomaly.autocorr(12):.2f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_annual_trend(annual, FIGURES_DIR / "weather_annual_trend.png")
    plot_seasonal_profile(monthly, FIGURES_DIR / "weather_seasonal_profile.png")
    print()
    print(f"Графики сохранены в {FIGURES_DIR}")
