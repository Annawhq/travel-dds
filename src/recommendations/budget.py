"""Этап 7: расчет сметы поездки и проверка бюджетных ограничений
Модуль не ищет цены билетов или проживания. Стоимости поступают
из пользовательского ввода или отдельного источника данных
Неизвестные стоимости нельзя подставлять как нулевые
"""
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.recommendations.profile import TravelProfile


def _check_amount(name: str, value: float) -> None:
    """Проверка денежных сумм: число должно быть конечным и >= 0"""
    if isinstance(value, bool):
        raise ValueError(f"{name}: ожидалось неотрицательное число")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: ожидалось неотрицательное число") from exc
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name}: ожидалось неотрицательное конечное число")


@dataclass(frozen=True)
class TripCosts:
    """Оценочные расходы на поездку, в рублях
    Все компоненты вводятся явно. Значение 0 означает отсутствие расходов,
    а не отсутствие данных о цене
    """
    round_trip_tickets: float       # Дорога туда и обратно
    price_per_night: float          # Цена одной ночи проживания
    nights: int                     # Число ночей
    activities_and_food: float      # Достопримечательности + питание
    local_transport: float          # Передвижение внутри города
    reserve_fraction: float = 0.10  # Резерв от суммы расходов (10%)

    def __post_init__(self) -> None:
        for name in (
            "round_trip_tickets", "price_per_night",
            "activities_and_food", "local_transport",
            "reserve_fraction",
        ):
            _check_amount(name, getattr(self, name))

        if (
            isinstance(self.nights, bool)
            or not isinstance(self.nights, int)
            or self.nights < 0
        ):
            raise ValueError(
                "nights должен быть неотрицательным целым числом"
            )

        if self.reserve_fraction > 1:
            raise ValueError(
                "reserve_fraction должен быть от 0 до 1"
            )


@dataclass(frozen=True)
class BudgetLimits:
    """Максимальные расходы; None означает отсутствие лимита"""
    total: float | None = None
    tickets: float | None = None
    activities_and_food: float | None = None

    def __post_init__(self) -> None:
        for name in ("total", "tickets", "activities_and_food"):
            value = getattr(self, name)
            if value is not None:
                _check_amount(name, value)

    @classmethod
    def from_profile(cls, profile: "TravelProfile") -> "BudgetLimits":
        """Берет ограничения из TravelProfile (шаг 1)."""
        return cls(
            total=profile.total_budget,
            tickets=profile.tickets_budget,
            activities_and_food=profile.activity_budget,
        )


def calculate_budget(costs: TripCosts, limits: BudgetLimits | None = None,) -> dict:
    """Вычисляет смету и сообщает, какие лимиты нарушены"""
    if limits is None:
        limits = BudgetLimits()

    # 1. Стоимость проживания
    accommodation = costs.price_per_night * costs.nights
    # 2. Основные расходы
    subtotal = (
        costs.round_trip_tickets
        + accommodation
        + costs.activities_and_food
        + costs.local_transport
    )
    # 3. Резерв
    reserve = subtotal * costs.reserve_fraction
    # 4. Полная стоимость
    total = subtotal + reserve
    # 5. Проверяем ограничения пользователя
    checks = {
        "общий бюджет": (
            limits.total is None or total <= limits.total
        ),
        "бюджет на билеты": (
            limits.tickets is None
            or costs.round_trip_tickets <= limits.tickets
        ),
        "бюджет на объекты и питание": (
            limits.activities_and_food is None
            or costs.activities_and_food <= limits.activities_and_food
        ),
    }

    return {
        "tickets": costs.round_trip_tickets,
        "accommodation": accommodation,
        "activities_and_food": costs.activities_and_food,
        "local_transport": costs.local_transport,
        "reserve": reserve,
        "total": total,
        "fits_limits": all(checks.values()),
        "violations": [
            name for name, passed in checks.items()
            if not passed
        ],
        "remaining_total": (
            None if limits.total is None
            else limits.total - total
        ),
    }


if __name__ == "__main__":
    from src.recommendations.profile import TravelProfile
    # Демонстрационные цены на поездку
    costs = TripCosts(
        round_trip_tickets=8000,
        price_per_night=3500,
        nights=1,
        activities_and_food=3000,
        local_transport=800,
        reserve_fraction=0.10,
    )

    profile = TravelProfile(
        total_budget=17000,
        tickets_budget=9000,
        activity_budget=3000,
    )

    result = calculate_budget(
        costs,
        BudgetLimits.from_profile(profile),
    )

    for label, key in (
        ("Билеты", "tickets"),
        ("Проживание", "accommodation"),
        ("Объекты и питание", "activities_and_food"),
        ("Городской транспорт", "local_transport"),
        ("Резерв", "reserve"),
        ("Итого", "total"),
    ):
        print(f"{label}: {result[key]:,.2f} руб.")

    print("Лимиты соблюдены:", result["fits_limits"])
    print(
        "Остаток общего бюджета:",
        round(result["remaining_total"], 2),
        "руб.",
    )
    print("Нарушения:", result["violations"])
