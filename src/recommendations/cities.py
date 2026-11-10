
"""Этап 7: персонализированное ранжирование туристических городов

Входные данные в data/test/ пока экспертные и тестовые. Фактические
цены проезда, размещения и привязка к городу отправления здесь не учтены
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.multicriteria.ahp import evaluate_alternatives
from src.multicriteria.alternatives import (
    CRITERIA_COLUMNS,
    PROJECT_ROOT,
    load_alternatives,
)
from src.uncertainty.criteria import bayes, find_dominance, hurwicz, risk_std, wald
from src.uncertainty.scenarios import (
    build_payoff_matrix,
    build_scenarios,
    load_uncertainty,
)
from src.recommendations.profile import THEMES, TravelProfile


DESTINATIONS_PATH = PROJECT_ROOT / "data" / "test" / "destinations.csv"
THEMES_PATH = PROJECT_ROOT / "data" / "test" / "city_themes.csv"
UNCERTAINTY_PATH = PROJECT_ROOT / "data" / "test" / "uncertainty.csv"


def load_city_themes(path: Path = THEMES_PATH) -> pd.DataFrame:
    """Возвращает таблицу оценок тематик с индексом по названию города"""
    df = pd.read_csv(path)
    columns = list(THEMES.values())
    required = ["city", *columns]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"В city_themes.csv отсутствуют столбцы: {sorted(missing)}")
    if df["city"].isna().any() or df["city"].duplicated().any():
        raise ValueError("Названия городов должны быть заполнены и уникальны")

    values = df[columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not values.ge(1).all().all() or not values.le(5).all().all():
        raise ValueError("Тематические оценки должны быть числами от 1 до 5")
    return values.set_axis(df["city"].tolist(), axis="index")


def recommend_cities(profile: TravelProfile, top_n: int | None = 5) -> pd.DataFrame:
    """Ранжирует направления по интересам, весам критериев и отношению к риску"""
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
        raise ValueError("top_n должен быть положительным целым числом или None")
    cities, scores = load_alternatives(DESTINATIONS_PATH)
    themes = load_city_themes()
    if set(cities) != set(themes.index):
        raise ValueError("Списки городов в destinations.csv и city_themes.csv не совпадают")
    # Персонализируем критерий interests
    scores = scores.copy()
    if profile.interests:
        selected = [THEMES[theme] for theme in profile.interests]
        interest_scores = themes.loc[cities, selected].mean(axis=1).to_numpy()
    else:
        # Нейтральная оценка при отсутствии интересов
        interest_scores = np.full(len(cities), 3.0)

    scores[:, CRITERIA_COLUMNS.index("interests")] = interest_scores
    weights = profile.criterion_weights()
    baseline = evaluate_alternatives(scores, weights)

    # Сценарии неопределенности из этапа 3
    scenarios = build_scenarios()
    uncertainty = load_uncertainty(UNCERTAINTY_PATH, cities)
    payoff = build_payoff_matrix(cities, scores, uncertainty, scenarios, weights)
    probabilities = scenarios["probability"].to_numpy()

    # Метод выбора зависит от отношения пользователя к риску
    if profile.risk_attitude == "осторожный":
        result_scores = wald(payoff)
        method = "Вальд"
    elif profile.risk_attitude == "оптимист":
        result_scores = hurwicz(payoff, alpha=0.8)
        method = "Гурвиц (α=0.8)"
    else:
        result_scores = bayes(payoff, probabilities)
        method = "Байес"

    deviations = risk_std(payoff, probabilities)
    dominance = find_dominance(payoff)

    # Формируем результаты и объяснения
    rows = []
    for index, city in enumerate(cities):
        interest_note = ", ".join(profile.interests) if profile.interests else "не заданы"
        explanation = (
            f"Интересы ({interest_note}): {interest_scores[index]:.2f}/5; "
            f"выбор по критерию {method}: {result_scores[city]:.3f}/5; "
            f"риск (СКО): {deviations[city]:.3f}. "
            "Стоимость поездки и транспорт из города отправления пока не проверены."
        )
        rows.append({
            "city": city,
            "score": float(result_scores[city]),
            "base_score": float(baseline[index]),
            "interests_score": float(interest_scores[index]),
            "risk_std": float(deviations[city]),
            "dominated_by": len(dominance[city]),
            "method": method,
            "budget_status": "не проверен: нет сметы",
            "explanation": explanation,
        })

    result = pd.DataFrame(rows).sort_values(
        ["score", "city"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    return result if top_n is None else result.head(top_n).copy()


if __name__ == "__main__":
    # Два пользователя с разными интересами и одинаковыми весами
    interest_priority = {
        "cost": 0.05,
        "weather": 0.05,
        "transport": 0.05,
        "attractiveness": 0.15,
        "interests": 0.70,
    }

    for theme in ("Культура и история", "Прогулки и природа"):
        tourist = TravelProfile(
            interests=(theme,),
            weights=interest_priority,
        )
        ranking = recommend_cities(tourist, top_n=5)

        print(f"\nИнтерес: {theme} (тестовые оценки)")
        print(
            ranking[["city", "score", "interests_score"]]
            .round(3)
            .to_string(index=False)
        )
