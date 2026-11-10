
"""Профиль туриста (ЛПР) для рекомендательного модуля этапа 7"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping
import numpy as np
from src.knowledge.facts import TAXONOMY
from src.multicriteria.ahp import TEST_COMPARISON_MATRIX, calculate_weights
from src.multicriteria.alternatives import CRITERIA_COLUMNS


# Тематики совпадают с таксономией базы знаний этапа 5
THEMES = {
    "Культура и история": "culture",
    "Прогулки и природа": "nature",
    "Гастрономия": "food",
    "Развлечения": "entertainment",
    "Загородные поездки": "excursions",
}

if set(THEMES) != set(TAXONOMY.values()):
    raise RuntimeError("Тематики этапов 5 и 7 не совпадают")

RISK_ATTITUDES = {"осторожный", "нейтральный", "оптимист"}
WEATHER = {"лучше", "обычная", "хуже"}


@dataclass(frozen=True)
class TravelProfile:
    """Параметры поездки, которые задает ЛПР
    Бюджеты указаны в рублях; None означает отсутствие отдельного лимита
    weights=None означает использование исходных весов AHP этапа 2
    """
    interests: tuple[str, ...] = ()
    origin: str | None = None
    days: int = 2
    total_budget: float | None = None
    tickets_budget: float | None = None
    activity_budget: float | None = 3000.0
    with_children: bool = False
    first_visit: bool = True
    weather: str = "обычная"
    risk_attitude: str = "нейтральный"
    weights: Mapping[str, float] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.days, bool) or not isinstance(self.days, int) or self.days < 1:
            raise ValueError("days должен быть положительным целым числом")

        if self.origin is not None and not self.origin.strip():
            raise ValueError("origin не может быть пустой строкой")

        unknown = set(self.interests) - set(THEMES)
        if unknown:
            raise ValueError(f"Неизвестные тематики: {sorted(unknown)}")

        if self.weather not in WEATHER:
            raise ValueError("weather: 'лучше', 'обычная' или 'хуже'")
        if self.risk_attitude not in RISK_ATTITUDES:
            raise ValueError("risk_attitude: осторожный/нейтральный/оптимист")

        for name in ("total_budget", "tickets_budget", "activity_budget"):
            value = getattr(self, name)
            if value is not None:
                try:
                    amount = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} должен быть числом") from exc
                if not isfinite(amount) or amount < 0:
                    raise ValueError(f"{name} должен быть неотрицательным числом")

        if self.weights is not None:
            if set(self.weights) != set(CRITERIA_COLUMNS):
                raise ValueError(f"weights должны включать критерии: {CRITERIA_COLUMNS}")
            try:
                values = [float(self.weights[name]) for name in CRITERIA_COLUMNS]
            except (TypeError, ValueError) as exc:
                raise ValueError("Веса должны быть числами") from exc
            if any(not isfinite(value) or value < 0 for value in values):
                raise ValueError("Веса должны быть конечными и неотрицательными")
            if sum(values) <= 0:
                raise ValueError("Сумма весов должна быть положительной")

    def criterion_weights(self) -> np.ndarray:
        """Веса пяти критериев в порядке CRITERIA_COLUMNS; сумма равна 1"""
        if self.weights is None:
            return calculate_weights(TEST_COMPARISON_MATRIX)
        values = np.array([self.weights[name] for name in CRITERIA_COLUMNS], dtype=float)
        return values / values.sum()

    def to_knowledge_profile(self) -> dict:
        """Преобразует профиль к формату базы знаний этапа 5"""
        # В этапе 5 бюджет сравнивается с ценой ОДНОГО объекта
        # При отсутствии лимита используем технический большой порог
        return {
            "interests": list(self.interests),
            "budget": self.activity_budget if self.activity_budget is not None else 10**12,
            "days": self.days,
            "with_children": self.with_children,
            "first_visit": self.first_visit,
            "weather": self.weather,
            "risk_attitude": self.risk_attitude,
        }
