import numpy as np
import pandas as pd
from src.multicriteria.ahp import TEST_COMPARISON_MATRIX, calculate_weights
from src.multicriteria.alternatives import PROJECT_ROOT, load_alternatives
from src.uncertainty.scenarios import (
    build_payoff_matrix,
    build_scenarios,
    load_uncertainty,
)


def find_dominance(payoff: pd.DataFrame) -> dict[str, list[str]]:
    """
    Для каждой альтернативы возвращает список альтернатив, которые
    ее доминируют: не хуже во всех сценариях и лучше хотя бы в одном
    """
    dominated_by = {}
    for a in payoff.index:
        dominators = []
        for b in payoff.index:
            if a == b:
                continue
            not_worse = (payoff.loc[b] >= payoff.loc[a]).all()
            better = (payoff.loc[b] > payoff.loc[a]).any()
            if not_worse and better:
                dominators.append(b)
        dominated_by[a] = dominators
    return dominated_by


def wald(payoff: pd.DataFrame) -> pd.Series:
    """Максиминный критерий: гарантированный результат в худшем сценарии"""
    return payoff.min(axis=1)


def maximax(payoff: pd.DataFrame) -> pd.Series:
    """Максимаксный критерий: результат в лучшем сценарии"""
    return payoff.max(axis=1)


def savage(payoff: pd.DataFrame) -> pd.Series:
    """
    Критерий Сэвиджа: максимальное сожаление
    Сожаление = лучший результат в сценарии − результат альтернативы
    Чем меньше, тем лучше
    """
    regret = payoff.max(axis=0) - payoff
    return regret.max(axis=1)


def hurwicz(payoff: pd.DataFrame, alpha: float) -> pd.Series:
    """
    Критерий Гурвица: alpha * лучший + (1 − alpha) * худший результат
    alpha - коэффициент оптимизма ЛПР (0 - пессимист, 1 - оптимист)
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha должен быть в диапазоне [0, 1].")
    return alpha * payoff.max(axis=1) + (1 - alpha) * payoff.min(axis=1)


def laplace(payoff: pd.DataFrame) -> pd.Series:
    """Критерий Лапласа: все сценарии считаются равновероятными"""
    return payoff.mean(axis=1)


def bayes(payoff: pd.DataFrame, probabilities: np.ndarray) -> pd.Series:
    """Критерий Байеса: математическое ожидание с учетом вероятностей"""
    return payoff @ probabilities


def risk_std(payoff: pd.DataFrame, probabilities: np.ndarray) -> pd.Series:
    """Стандартное отклонение результата - мера риска альтернативы"""
    expected = payoff @ probabilities
    variance = ((payoff.sub(expected, axis=0)) ** 2) @ probabilities
    return np.sqrt(variance)


def summarize(
    payoff: pd.DataFrame,
    probabilities: np.ndarray,
    alpha: float = 0.5,) -> pd.DataFrame:
    """Сводная таблица значений всех критериев для каждой альтернативы"""
    return pd.DataFrame({
        "Вальд": wald(payoff),
        "Максимакс": maximax(payoff),
        f"Гурвиц a={alpha}": hurwicz(payoff, alpha),
        "Лаплас": laplace(payoff),
        "Байес": bayes(payoff, probabilities),
        "Сэвидж": savage(payoff),
        "Риск (СКО)": risk_std(payoff, probabilities),
    })


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    pd.set_option("display.max_columns", None)

    weights = calculate_weights(TEST_COMPARISON_MATRIX)
    cities, scores = load_alternatives(
        PROJECT_ROOT / "data" / "test" / "destinations.csv"
    )
    uncertainty = load_uncertainty(
        PROJECT_ROOT / "data" / "test" / "uncertainty.csv", cities
    )
    scenarios = build_scenarios()
    probabilities = scenarios["probability"].to_numpy()
    payoff = build_payoff_matrix(cities, scores, uncertainty, scenarios, weights)

    # 1. Анализ доминирования
    dominance = find_dominance(payoff)
    non_dominated = [c for c, d in dominance.items() if not d]
    print("Недоминируемые альтернативы:", non_dominated)

    # Повторяем анализ без лидера, чтобы найти кандидатов на второе место
    rest = payoff.drop(index=non_dominated)
    rest_dominance = find_dominance(rest)
    second_level = [c for c, d in rest_dominance.items() if not d]
    print("Недоминируемые без лидера:", second_level)
    print()

    # 2. Значения критериев
    summary = summarize(payoff, probabilities, alpha=0.5)
    print("Значения критериев:")
    print(summary)
    print()

    # 3. Топ-3 по каждому критерию
    print("Топ-3 по критериям:")
    for column in summary.columns:
        if column == "Риск (СКО)":
            continue
        ascending = column == "Сэвидж"  # для Сэвиджа меньше - лучше
        top = summary[column].sort_values(ascending=ascending).head(3)
        print(f"{column:>14}: " + ", ".join(top.index))
