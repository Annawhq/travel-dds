from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from src.multicriteria.ahp import (
    TEST_COMPARISON_MATRIX,
    calculate_weights,
    evaluate_alternatives,
)
from src.multicriteria.alternatives import (
    CRITERIA_COLUMNS,
    PROJECT_ROOT,
    load_alternatives,
)

COST_INDEX = CRITERIA_COLUMNS.index("cost")
WEATHER_INDEX = CRITERIA_COLUMNS.index("weather")

# Состояния неопределенных факторов: (название, сдвиг, вероятность)
# Сдвиг +1 - фактор лучше ожидаемого, -1 - хуже, 0 - базовый уровень
# Вероятности тестовые (экспертная оценка)
COST_STATES = [
    ("цены ниже", +1, 0.2),
    ("цены базовые", 0, 0.5),
    ("цены выше", -1, 0.3),
]
WEATHER_STATES = [
    ("погода лучше", +1, 0.3),
    ("погода обычная", 0, 0.5),
    ("погода хуже", -1, 0.2),
]


def build_scenarios() -> pd.DataFrame:
    """
    Формирует сценарии как все сочетания состояний стоимости и погоды
    Факторы считаются независимыми: P(сценария) = P(цены) * P(погоды)
    """
    rows = []
    for number, (cost, weather) in enumerate(
        product(COST_STATES, WEATHER_STATES), start=1
    ):
        rows.append({
            "scenario": f"S{number}",
            "cost_state": cost[0],
            "weather_state": weather[0],
            "cost_shift": cost[1],
            "weather_shift": weather[1],
            "probability": cost[2] * weather[2],
        })
    scenarios = pd.DataFrame(rows)

    if not np.isclose(scenarios["probability"].sum(), 1.0):
        raise ValueError("Сумма вероятностей сценариев не равна 1.")
    return scenarios


def load_uncertainty(path: Path, cities: list[str]) -> pd.DataFrame:
    """
    Загружает чувствительность городов к неопределенным факторам
    (на сколько баллов шкалы 1-5 меняется оценка при сдвиге фактора)
    """
    df = pd.read_csv(path).set_index("city")
    missing = set(cities) - set(df.index)
    if missing:
        raise ValueError(f"Нет параметров неопределенности для: {missing}")
    if (df < 0).any().any():
        raise ValueError("Чувствительность не может быть отрицательной.")
    return df.loc[cities]


def build_payoff_matrix(
    cities: list[str],
    scores: np.ndarray,
    uncertainty: pd.DataFrame,
    scenarios: pd.DataFrame,
    weights: np.ndarray,) -> pd.DataFrame:
    """
    Матрица выигрышей: итоговая оценка каждого города в каждом сценарии
    Оценки стоимости и погоды сдвигаются и обрезаются до шкалы [1, 5],
    затем пересчитывается взвешенная сумма с весами AHP
    """
    payoff = pd.DataFrame(index=cities, columns=scenarios["scenario"], dtype=float)

    cost_vol = uncertainty["cost_volatility"].to_numpy()
    weather_sens = uncertainty["weather_sensitivity"].to_numpy()

    for _, s in scenarios.iterrows():
        scenario_scores = scores.copy()
        scenario_scores[:, COST_INDEX] += s["cost_shift"] * cost_vol
        scenario_scores[:, WEATHER_INDEX] += s["weather_shift"] * weather_sens
        scenario_scores = np.clip(scenario_scores, 1, 5)
        payoff[s["scenario"]] = evaluate_alternatives(scenario_scores, weights)

    return payoff


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)

    weights = calculate_weights(TEST_COMPARISON_MATRIX)
    cities, scores = load_alternatives(
        PROJECT_ROOT / "data" / "test" / "destinations.csv"
    )
    uncertainty = load_uncertainty(
        PROJECT_ROOT / "data" / "test" / "uncertainty.csv", cities
    )
    scenarios = build_scenarios()

    print("Сценарии:")
    print(scenarios[["scenario", "cost_state", "weather_state", "probability"]]
          .to_string(index=False))
    print(f"Сумма вероятностей: {scenarios['probability'].sum():.2f}")
    print()

    payoff = build_payoff_matrix(cities, scores, uncertainty, scenarios, weights)
    print("Матрица выигрышей (итоговая оценка города в сценарии):")
    print(payoff)
