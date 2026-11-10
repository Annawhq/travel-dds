"""Этап 7: проверка доступности городов по смете путешествия
Стоимость не загружается из интернета. Источник данных о расходах передает
вызывающий код. Отсутствующая смета не считается нулевой стоимостью
"""
from dataclasses import dataclass
from datetime import date
from typing import Iterable
import pandas as pd
from src.recommendations.budget import (
    BudgetLimits,
    TripCosts,
    calculate_budget,
)
from src.recommendations.cities import recommend_cities
from src.recommendations.profile import TravelProfile


@dataclass(frozen=True)
class CityCostEstimate:
    """Смета для пары «город отправления - город назначения»
    days - длительность поездки, costs - рассчитанные/введенные расходы
    source должен описывать происхождение цен (включая пометку «тестовые»)
    checked_on - дата проверки цен, если она известна
    """
    origin: str
    city: str
    days: int
    costs: TripCosts
    source: str
    checked_on: date | None = None

    def __post_init__(self) -> None:
        for field_name in ("origin", "city", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} должен быть непустой строкой"
                )

        if (
            isinstance(self.days, bool)
            or not isinstance(self.days, int)
            or self.days < 1
        ):
            raise ValueError(
                "days должен быть положительным целым числом"
            )

        if not isinstance(self.costs, TripCosts):
            raise TypeError(
                "costs должен быть экземпляром TripCosts"
            )

        if (
            self.checked_on is not None
            and not isinstance(self.checked_on, date)
        ):
            raise TypeError(
                "checked_on должен быть датой или None"
            )


def recommend_cities_with_budget(
    profile: TravelProfile,
    estimates: Iterable[CityCostEstimate],
    top_n: int | None = 5,
    only_affordable: bool = True,) -> pd.DataFrame:
    """Дополняет рейтинг городов проверкой стоимости путешествия
    only_affordable=True: возвращает только города с известной сметой,
    которая удовлетворяет всем установленным лимитам
    only_affordable=False: возвращает все города с отметками:
    «в пределах лимитов», «превышение бюджета», «смета отсутствует»
    Рейтинг AHP/риска не пересчитывается - это отдельная проверка
    """
    if profile.origin is None:
        raise ValueError(
            "Укажи origin в TravelProfile для проверки стоимости"
        )

    if top_n is not None and (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or top_n < 1
    ):
        raise ValueError(
            "top_n должен быть положительным целым числом или None"
        )
    # Получаем все 15 городов, а не только первые пять
    ranking = (
        recommend_cities(profile, top_n=None)
        .drop(columns=["budget_status"])
        .copy()
    )
    known_cities = set(ranking["city"])
    offers: dict[str, CityCostEstimate] = {}
    # Выбираем сметы только для города отправления
    # и продолжительности поездки текущего пользователя
    for estimate in estimates:
        if not isinstance(estimate, CityCostEstimate):
            raise TypeError(
                "estimates должны содержать CityCostEstimate"
            )

        if (
            estimate.origin != profile.origin
            or estimate.days != profile.days
        ):
            continue

        if estimate.city not in known_cities:
            raise ValueError(
                f"Нет такого города в рейтинге: {estimate.city}"
            )

        if estimate.city in offers:
            raise ValueError(
                f"Несколько смет для города: {estimate.city}"
            )

        offers[estimate.city] = estimate
    limits = BudgetLimits.from_profile(profile)
    details = []
    budget_explanations = []
    # Проверяем финансовую доступность каждого города
    for city in ranking["city"]:
        estimate = offers.get(city)
        if estimate is None:
            budget_explanations.append(
                f"Для поездки {profile.origin} — {city} "
                f"на {profile.days} дн. нет данных о стоимости."
            )

            details.append({
                "estimated_total": None,
                "tickets": None,
                "budget_ok": None,
                "budget_status": "смета отсутствует",
                "violations": "",
                "price_source": "",
                "checked_on": None,
            })
            continue

        calculation = calculate_budget(
            estimate.costs,
            limits,
        )
        ok = calculation["fits_limits"]
        limit_note = (
            "Лимиты соблюдены."
            if ok
            else (
                "Нарушения: "
                + "; ".join(calculation["violations"])
                + "."
            )
        )
        budget_explanations.append(
            f"Оценочная стоимость: "
            f"{calculation['total']:,.0f} руб. "
            f"Источник: {estimate.source}. {limit_note}"
        )
        details.append({
            "estimated_total": calculation["total"],
            "tickets": calculation["tickets"],
            "budget_ok": ok,
            "budget_status": (
                "в пределах лимитов"
                if ok
                else "превышение бюджета"
            ),
            "violations": "; ".join(
                calculation["violations"]
            ),
            "price_source": estimate.source,
            "checked_on": estimate.checked_on,
        })
    # Добавляем результаты к персональному рейтингу городов
    details_df = pd.DataFrame(details)
    result = pd.concat(
        [ranking, details_df],
        axis=1,
    )
    # Обновляем объяснения с учетом выполненной проверки
    outdated_note = (
        "Стоимость поездки и транспорт из города "
        "отправления пока не проверены."
    )
    result["explanation"] = [
        original.replace(
            outdated_note,
            "Оценка транспортной доступности пока тестовая."
        ) + " " + budget_note
        for original, budget_note in zip(
            result["explanation"],
            budget_explanations,
        )
    ]
    # При необходимости оставляем только доступные города
    if only_affordable:
        result = result[
            result["budget_ok"] == True
        ].copy()
    result = result.reset_index(drop=True)
    return (
        result
        if top_n is None
        else result.head(top_n).copy()
    )


if __name__ == "__main__":
    # Демонстрационные данные
    profile = TravelProfile(
        origin="Москва",
        interests=("Культура и история",),
        days=2,
        total_budget=18_000,
        tickets_budget=9_000,
        activity_budget=3_000,
    )

    sample = [
        CityCostEstimate(
            origin="Москва",
            city="Казань",
            days=2,
            costs=TripCosts(
                8_000, 3_500, 1, 3_000, 800
            ),
            source="тестовый пример, не рыночная цена",
        ),
        CityCostEstimate(
            origin="Москва",
            city="Калининград",
            days=2,
            costs=TripCosts(
                15_000, 4_000, 1, 3_000, 1_000
            ),
            source="тестовый пример, не рыночная цена",
        ),
    ]
    # Показываем статусы трех направлений
    all_cities = recommend_cities_with_budget(
        profile,
        sample,
        top_n=None,
        only_affordable=False,
    )

    print(
        "Статусы трех городов "
        "(только демонстрационные сметы):"
    )
    selected = all_cities.set_index("city").loc[
        ["Казань", "Калининград", "Сочи"],
        ["score", "estimated_total", "budget_status"],
    ]
    print(selected.round(2).to_string())
    # Показываем только города, прошедшие проверку
    affordable = recommend_cities_with_budget(
        profile,
        sample,
    )
    print(
        "\nПодходящие по предоставленным сметам города:"
    )
    print(
        affordable[
            ["city", "estimated_total"]
        ].to_string(index=False)
    )
