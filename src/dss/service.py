"""Этап 8. Управление рекомендациями, сметой и маршрутом по Казани
Для выбора городов используются этапы 2, 3 и 7, для рекомендаций
достопримечательностей - этапы 5 и 7, для маршрута - оптимизатор этапа 4.
Прогноз этапа 6 подключается отдельно. Цены в data/test/ - учебные.
"""
from dataclasses import dataclass, replace
from math import inf
from pathlib import Path

import pandas as pd

from src.knowledge.facts import TOURIST, add_profile
from src.knowledge.inference import forward_chain
from src.knowledge.kb import KnowledgeBase
from src.knowledge.rules import RULES
from src.multicriteria.alternatives import PROJECT_ROOT
from src.optimization.objects import load_objects
from src.optimization.planner import plan_with_feedback
from src.optimization.travel import cluster_access_time, cluster_objects
from src.optimization.trip_plan import BASE_ID
from src.recommendations.budget import BudgetLimits, calculate_budget
from src.recommendations.city_budget import CityCostEstimate, recommend_cities_with_budget
from src.recommendations.cost_catalog import load_cost_estimates
from src.recommendations.places import KAZAN_CATALOG, recommend_places
from src.recommendations.profile import TravelProfile

DEFAULT_COSTS_PATH = PROJECT_ROOT / "data" / "test" / "city_costs.csv"


@dataclass
class RouteResult:
    """Выбранные объекты, расписания и стоимость готового маршрута"""
    plan: pd.DataFrame
    schedules: dict[int, pd.DataFrame]
    history: pd.DataFrame
    meals_per_day: int
    activity_limit: float
    activities_cost: float
    personal_score: float
    final_budget: dict


@dataclass
class DecisionResult:
    all_cities: pd.DataFrame
    affordable_cities: pd.DataFrame
    selected_city: str | None
    city_budget: dict | None
    places: pd.DataFrame
    notices: list[str]
    route: RouteResult | None = None


def _meals_from_knowledge_base(profile: TravelProfile) -> int:
    """Правила R10/R11 задают 2 приема пищи для семей и гурманов"""
    kb = KnowledgeBase()
    add_profile(kb, profile.to_knowledge_profile())
    route_rules = [rule for rule in RULES if rule.name in ("R10", "R11")]
    forward_chain(kb, route_rules)
    values = [
        value for subject, relation, value in kb.facts
        if subject == TOURIST and relation == "приемов_пищи_в_день"
    ]
    return max(values, default=1)


def _activity_limit(profile: TravelProfile, estimate: CityCostEstimate) -> float:
    """Максимум на объекты после резервирования дороги, жилья и транспорта
    Необязательные лимиты не ограничивают расход; сумма стоимости объектов
    каталога используется только как конечная верхняя граница при их отсутствии
    """
    costs = estimate.costs
    limits = []
    if profile.activity_budget is not None:
        limits.append(float(profile.activity_budget))
    if profile.total_budget is not None:
        fixed = (
            costs.round_trip_tickets + costs.price_per_night * costs.nights
            + costs.local_transport
        )
        # Резерв начисляется на все расходы, включая объекты и питание
        limits.append(max(0.0, profile.total_budget / (1 + costs.reserve_fraction) - fixed))
    return min(limits) if limits else inf


def _build_kazan_route(
    profile: TravelProfile,
    estimate: CityCostEstimate,
    ranked_places: pd.DataFrame,
    feedback_extra_min: int,) -> RouteResult:
    """Отбор объектов по персональным баллам + реальный порядок посещения
    Каталог этапа 4 и координаты гостиницы не меняем. Используем весь
    допустимый набор объектов, а не только показанные пользователю Top-10
    """
    meals = _meals_from_knowledge_base(profile)
    budget_cap = _activity_limit(profile, estimate)

    objects = load_objects(KAZAN_CATALOG).set_index("id")
    hotel = objects.loc[BASE_ID]  # условная гостиница возле Кремля
    objects["cluster"] = cluster_objects(objects)

    scores = ranked_places.set_index("object_id")["score"]
    objects = objects.loc[objects.index.intersection(scores.index)].copy()
    if objects.empty:
        raise RuntimeError("После фильтрации не осталось доступных объектов")
    # Вместо исходной популярности этапа 4 оптимизируем полезность этапа 7
    objects["original_attractiveness"] = objects["attractiveness"]
    objects["attractiveness"] = scores.loc[objects.index].astype(float)

    if (objects["category"] == "food").sum() < meals * profile.days:
        raise RuntimeError("Не хватает объектов питания для заданного числа приемов пищи")

    if budget_cap == inf:
        budget_cap = float(objects["price"].sum())

    cluster_access = cluster_access_time(
        objects, objects["cluster"], float(hotel["lat"]), float(hotel["lon"])
    )
    trip = {
        "days": profile.days,
        "day_minutes": 600,  # 10:00–20:00, как в апробированной модели этапа 4
        "budget": budget_cap,
        "meals_per_day": meals,
        "max_per_category": 2,
    }
    best, history = plan_with_feedback(
        objects, cluster_access, float(hotel["lat"]), float(hotel["lon"]),
        trip, max_extra=feedback_extra_min,
    )
    if best is None:
        raise RuntimeError("Не удалось построить маршрут в пределах часов работы и дня")

    _time_budget, plan, schedules = best
    cost_of_places = float(plan["price"].sum())
    adjusted = replace(estimate.costs, activities_and_food=cost_of_places)
    final_budget = calculate_budget(adjusted, BudgetLimits.from_profile(profile))
    if not final_budget["fits_limits"]:
        raise RuntimeError("Маршрут нарушает бюджетные ограничения")

    plan = plan.copy()
    plan["object_id"] = plan.index
    plan["personal_score"] = plan["attractiveness"]
    return RouteResult(
        plan=plan,
        schedules=schedules,
        history=history,
        meals_per_day=meals,
        activity_limit=round(budget_cap, 2),
        activities_cost=cost_of_places,
        personal_score=round(float(plan["personal_score"].sum()), 2),
        final_budget=final_budget,
    )


def analyze_trip(
    profile: TravelProfile,
    selected_city: str | None = None,
    top_cities: int = 5,
    top_places: int = 10,
    costs_path: str | Path = DEFAULT_COSTS_PATH,
    build_route: bool = False,
    feedback_extra_min: int = 20,) -> DecisionResult:
    """Города → смета → объекты → (по запросу) двухдневный маршрут Казани"""
    if profile.origin is None:
        raise ValueError("Укажи город отправления в TravelProfile.origin")
    for name, value in (("top_cities", top_cities), ("top_places", top_places)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} должен быть положительным целым числом")
    if (isinstance(feedback_extra_min, bool) or not isinstance(feedback_extra_min, int)
            or not 0 <= feedback_extra_min <= 120):
        raise ValueError("feedback_extra_min должен быть целым числом от 0 до 120")

    estimates = load_cost_estimates(costs_path)
    all_cities = recommend_cities_with_budget(
        profile, estimates, top_n=None, only_affordable=False
    )
    affordable = (
        all_cities.loc[all_cities["budget_ok"].eq(True)]
        .head(top_cities).copy().reset_index(drop=True)
    )
    notices = [
        "Цены и тематические оценки в data/test/ учебные, "
        "а не актуальные предложения перевозчиков и гостиниц."
    ]
    if all_cities["budget_ok"].isna().all():
        notices.append(
            f"Нет смет для отправления из {profile.origin} на {profile.days} дн."
        )
    elif affordable.empty:
        notices.append("Нет направлений, проходящих все бюджетные лимиты.")

    budget = None
    places = pd.DataFrame()
    route = None

    if selected_city is not None:
        if selected_city not in set(all_cities["city"]):
            raise ValueError(f"Неизвестный город назначения: {selected_city}")
        selected_row = all_cities.set_index("city").loc[selected_city]
        estimate = next(
            (e for e in estimates if e.origin == profile.origin
             and e.city == selected_city and e.days == profile.days),
            None,
        )
        if estimate is not None:
            budget = calculate_budget(
                estimate.costs, BudgetLimits.from_profile(profile)
            )
            budget["source"] = estimate.source
            budget["checked_on"] = estimate.checked_on

        financially_allowed = selected_row["budget_status"] == "в пределах лимитов"
        if not financially_allowed:
            notices.append(
                f"{selected_city}: {selected_row['budget_status']}. "
                "Финансовая допустимость поездки не подтверждена."
            )

        if selected_city == "Казань":
            all_places = recommend_places("Казань", profile, top_n=None)
            places = all_places.head(top_places).copy()
            if build_route:
                if profile.days != 2:
                    notices.append(
                        "Маршрут пока проверен только для двухдневной поездки."
                    )
                elif not financially_allowed or estimate is None:
                    notices.append(
                        "Маршрут не строится: сначала нужна допустимая смета поездки."
                    )
                else:
                    try:
                        route = _build_kazan_route(
                            profile, estimate, all_places, feedback_extra_min
                        )
                    except RuntimeError as exc:
                        notices.append(f"Маршрут не построен: {exc}")
            else:
                notices.append(
                    "Показаны рекомендации объектов, не маршрут. "
                    "Для маршрута укажи build_route=True."
                )
        else:
            notices.append(
                f"Для города {selected_city} пока нет каталога объектов; "
                "маршрут доступен только для Казани."
            )

    return DecisionResult(
        all_cities=all_cities,
        affordable_cities=affordable,
        selected_city=selected_city,
        city_budget=budget,
        places=places,
        notices=notices,
        route=route,
    )


if __name__ == "__main__":
    profile = TravelProfile(
        origin="Москва", days=2, total_budget=30_000,
        tickets_budget=15_000, activity_budget=3_000,
        interests=("Культура и история",), risk_attitude="нейтральный",
    )
    result = analyze_trip(profile, selected_city="Казань", build_route=True)
    print("Топ финансово допустимых городов:")
    print(result.affordable_cities[
        ["city", "score", "estimated_total"]
    ].round(2).to_string(index=False))
    print("\nКазань: предварительная смета:",
          f"{result.city_budget['total']:,.0f} руб.")
    if result.route is not None:
        print("\nМаршрут построен!")
        print("Приемов пищи в день:", result.route.meals_per_day)
        print("Объектов:", len(result.route.plan))
        print("Стоимость посещений и питания:", result.route.activities_cost)
        print("Оценка маршрута:", result.route.personal_score)
        print("Смета с фактическими объектами:",
              round(result.route.final_budget["total"], 2))
        for day, schedule in result.route.schedules.items():
            print(f"\nДень {day}:")
            print(schedule.to_string(index=False))
    else:
        print("\nМаршрут не сформирован")
    for note in result.notices:
        print("-", note)
