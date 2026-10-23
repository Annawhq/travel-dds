import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from src.multicriteria.ahp import TEST_COMPARISON_MATRIX, calculate_weights
from src.multicriteria.alternatives import PROJECT_ROOT, load_alternatives
from src.uncertainty.criteria import hurwicz
from src.uncertainty.decision_tree import (
    TEST_TICKET_PARAMS,
    build_ticket_tree,
    roll_back,
)
from src.uncertainty.scenarios import (
    build_payoff_matrix,
    build_scenarios,
    load_uncertainty,
)
matplotlib.use("Agg")  # сохраняем графики в файлы без открытия окна
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# 1. Чувствительность рейтинга к коэффициенту оптимизма

def ranking_by_alpha(payoff: pd.DataFrame, alpha: float) -> list[str]:
    """Рейтинг городов по критерию Гурвица при заданном alpha"""
    return hurwicz(payoff, alpha).sort_values(ascending=False).index.tolist()


def alpha_switch_points(
    payoff: pd.DataFrame,
    position: int,
    grid_size: int = 101,) -> list[tuple[float, str, str]]:
    """
    Находит значения alpha, при которых меняется город на заданном месте
    рейтинга (position = 1 - первое место, 2 - второе и т.д.)
    Грубый поиск по сетке + уточнение точки корнем уравнения
    H_a(alpha) − H_b(alpha) = 0 методом Брента
    """
    alphas = np.linspace(0, 1, grid_size)
    holders = [ranking_by_alpha(payoff, a)[position - 1] for a in alphas]

    switches = []
    for i in range(1, grid_size):
        old, new = holders[i - 1], holders[i]
        if old != new:
            def diff(a):
                return (hurwicz(payoff.loc[[old, new]], a).loc[old]
                        - hurwicz(payoff.loc[[old, new]], a).loc[new])
            root = brentq(diff, alphas[i - 1], alphas[i])
            switches.append((root, old, new))
    return switches


# 2. Чувствительность дерева решений к вероятностям

def strategy_costs(params: dict) -> dict[str, float]:
    """Ожидаемые расходы каждой стратегии первого уровня дерева"""
    tree = build_ticket_tree(params)
    roll_back(tree)
    return {tree.edges["D0", child]["label"]: tree.nodes[child]["value"]
            for child in tree.successors("D0")}


def with_plans_change(params: dict, p: float) -> dict:
    """Копия параметров с другой вероятностью срыва поездки"""
    new = copy.deepcopy(params)
    new["p_plans_change"] = p
    return new


def with_price_rise(params: dict, q: float) -> dict:
    """
    Копия параметров с другой вероятностью подорожания q
    Вероятность снижения цен сохраняется, базовое состояние
    получает остаток, чтобы сумма вероятностей оставалась равной 1
    """
    new = copy.deepcopy(params)
    (low_name, low_mult, p_low), (base_name, base_mult, _), \
        (high_name, high_mult, _) = new["price_change"]
    if not 0 <= q <= 1 - p_low:
        raise ValueError(f"q должна быть в диапазоне [0, {1 - p_low}].")
    new["price_change"] = [
        (low_name, low_mult, p_low),
        (base_name, base_mult, 1 - p_low - q),
        (high_name, high_mult, q),
    ]
    return new


def break_even(params, modifier, strategy_a, strategy_b, low, high):
    """
    Пороговое значение параметра, при котором две стратегии имеют одинаковые ожидаемые расходы
    Возвращает None, если на отрезке точки безубыточности нет
    """
    def diff(x):
        costs = strategy_costs(modifier(params, x))
        return costs[strategy_a] - costs[strategy_b]

    if diff(low) * diff(high) > 0:
        return None
    return brentq(diff, low, high)


# Графики

def plot_hurwicz(payoff: pd.DataFrame, cities: list[str], path) -> None:
    alphas = np.linspace(0, 1, 101)
    plt.figure(figsize=(8, 5))
    for city in cities:
        values = [hurwicz(payoff.loc[[city]], a).iloc[0] for a in alphas]
        plt.plot(alphas, values, label=city)
    plt.xlabel("Коэффициент оптимизма α")
    plt.ylabel("Оценка по критерию Гурвица")
    plt.title("Чувствительность рейтинга к отношению ЛПР к риску")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_plans_change(params: dict, path) -> None:
    ps = np.linspace(0, 0.4, 81)
    curves = {}
    for p in ps:
        for name, cost in strategy_costs(with_plans_change(params, p)).items():
            curves.setdefault(name, []).append(cost)
    plt.figure(figsize=(8, 5))
    for name, values in curves.items():
        plt.plot(ps, values, label=name)
    plt.xlabel("Вероятность срыва поездки")
    plt.ylabel("Ожидаемые расходы, руб.")
    plt.title("Чувствительность стратегии покупки билета")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    weights = calculate_weights(TEST_COMPARISON_MATRIX)
    cities, scores = load_alternatives(
        PROJECT_ROOT / "data" / "test" / "destinations.csv"
    )
    uncertainty = load_uncertainty(
        PROJECT_ROOT / "data" / "test" / "uncertainty.csv", cities
    )
    scenarios = build_scenarios()
    payoff = build_payoff_matrix(cities, scores, uncertainty, scenarios, weights)

    print("1. Критерий Гурвица: смена лидеров при изменении α")
    for position in (1, 2):
        switches = alpha_switch_points(payoff, position)
        start = ranking_by_alpha(payoff, 0.0)[position - 1]
        print(f"  Место {position}: при α = 0 — {start}")
        if not switches:
            print("    смен нет на всем отрезке [0, 1]")
        for alpha, old, new in switches:
            print(f"    α = {alpha:.3f}: {old} → {new}")
    print()

    params = TEST_TICKET_PARAMS
    print("2. Дерево решений: пороговые значения параметров")
    print(f"  Базовые расходы: {strategy_costs(params)}")

    pairs = [
        ("Подождать", "Невозвратный сейчас"),
        ("Возвратный сейчас", "Невозвратный сейчас"),
        ("Возвратный сейчас", "Подождать"),
    ]
    print("  Вероятность срыва поездки p (база 0.15):")
    for a, b in pairs:
        x = break_even(params, with_plans_change, a, b, 0.0, 0.99)
        text = "не пересекаются на [0, 0.99]" if x is None else f"p = {x:.4f}"
        print(f"    {a} = {b}: {text}")

    print("  Вероятность подорожания q (база 0.30):")
    x = break_even(params, with_price_rise,
                   "Подождать", "Невозвратный сейчас", 0.0, 0.8)
    text = "не пересекаются на [0, 0.8]" if x is None else f"q = {x:.4f}"
    print(f"    Подождать = Невозвратный сейчас: {text}")
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top_cities = ["Казань", "Нижний Новгород", "Ярославль",
                  "Калининград", "Сочи"]
    plot_hurwicz(payoff, top_cities, FIGURES_DIR / "hurwicz_alpha.png")
    plot_plans_change(params, FIGURES_DIR / "tree_plans_change.png")
    print(f"Графики сохранены в {FIGURES_DIR}")
