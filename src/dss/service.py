"""Этап 8. Единый сервис выбора города, бюджета и персонального маршрута.
Маршруты строятся для городов с каталогом объектов в формате этапа 4.
Тестовые сметы и тематические оценки НЕ являются рыночными данными.
Пока проверенный сценарий маршрутизации: два дня, 10:00–20:00
"""

from dataclasses import dataclass, replace
from math import inf, isfinite
from pathlib import Path
from typing import Mapping

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


@dataclass(frozen=True)
class CityRouteConfig:
    """Настройки маршрута для города с каталогом достопримечательностей
    city_center - центр проверки координат (широта, долгота)
    start_coordinates - условное место проживания, НЕ реальная гостиница
    start_object_id - альтернатива координатам: объект каталога у места
    проживания. В старом сценарии Казани используется объект 'kremlin'
    """
    catalog_path: str | Path
    city_center: tuple[float, float] | None = None
    start_coordinates: tuple[float, float] | None = None
    start_object_id: str | None = None
    max_distance_km: float = 60.0

    def __post_init__(self) -> None:
        if (self.start_coordinates is None) == (self.start_object_id is None):
            raise ValueError(
                "Нужно указать либо start_coordinates, либо start_object_id"
            )


# В реальных каталогах ниже данные еще предстоит собрать. Центры городов
# используются также как условные места проживания
# Добавление новых городов не требует менять алгоритмы: достаточно
# зарегистрировать CityRouteConfig и положить CSV в указанную папку
CITY_ROUTE_CONFIGS: dict[str, CityRouteConfig] = {
    "Казань": CityRouteConfig(
        catalog_path=KAZAN_CATALOG,
        start_object_id=BASE_ID,
    ),
    "Нижний Новгород": CityRouteConfig(
        catalog_path=PROJECT_ROOT / "data" / "raw" / "nizhny_novgorod_objects.csv",
        city_center=(56.3269, 44.0065),
        start_coordinates=(56.3269, 44.0065),
    ),
    "Калининград": CityRouteConfig(
        catalog_path=PROJECT_ROOT / "data" / "raw" / "kaliningrad_objects.csv",
        city_center=(54.7104, 20.4522),
        start_coordinates=(54.7104, 20.4522),
    ),
}


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
    """Правила R10/R11 задают два приема пищи для семей и гурманов"""
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
    """Допустимые расходы на объекты после дороги, проживания и резерва"""
    costs = estimate.costs
    limits = []
    if profile.activity_budget is not None:
        limits.append(float(profile.activity_budget))
    if profile.total_budget is not None:
        fixed = (
            costs.round_trip_tickets
            + costs.price_per_night * costs.nights
            + costs.local_transport
        )
        limits.append(
            max(0.0, profile.total_budget / (1 + costs.reserve_fraction) - fixed)
        )
    return min(limits) if limits else inf


def _start_point(
    config: CityRouteConfig, objects: pd.DataFrame
) -> tuple[float, float]:
    """Координаты начала/конца маршрутов всех дней"""
    if config.start_object_id is not None:
        if config.start_object_id not in objects.index:
            raise ValueError(
                f"В каталоге нет опорного объекта: {config.start_object_id}"
            )
        base = objects.loc[config.start_object_id]
        return float(base["lat"]), float(base["lon"])
    assert config.start_coordinates is not None
    return map_to_coordinates(config.start_coordinates)


def map_to_coordinates(point: tuple[float, float]) -> tuple[float, float]:
    """Преобразует пару координат к числовому формату"""
    try:
        lat, lon = map(float, point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Координаты места проживания должны быть (lat, lon)") from exc
    if not (isfinite(lat) and isfinite(lon)
            and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Некорректные координаты места проживания")
    return lat, lon


def _build_city_route(
    profile: TravelProfile,
    estimate: CityCostEstimate,
    ranked_places: pd.DataFrame,
    config: CityRouteConfig,
    feedback_extra_min: int,) -> RouteResult:
    """Персонализирует целевую функцию и строит маршрут заданного города"""
    meals = _meals_from_knowledge_base(profile)
    budget_cap = _activity_limit(profile, estimate)

    objects = load_objects(
        Path(config.catalog_path),
        city_center=config.city_center,
        max_distance_km=config.max_distance_km,
    ).set_index("id")
    hotel_lat, hotel_lon = _start_point(config, objects)

    if "hotel" in objects.index:
        raise ValueError("Идентификатор 'hotel' зарезервирован маршрутизатором")
    if len(objects) < 2:
        raise RuntimeError("Для кластеризации необходимо хотя бы два объекта")

    # Как в старой версии: кластеры вычисляются на всем каталоге до фильтров.
    objects["cluster"] = cluster_objects(objects)
    scores = ranked_places.set_index("object_id")["score"]
    objects = objects.loc[objects.index.intersection(scores.index)].copy()
    if objects.empty:
        raise RuntimeError("После фильтрации не осталось доступных объектов")

    # Оптимизатор этапа 4 максимизирует столбец attractiveness. Подставляем
    # в него персональные оценки, сохранив исходную привлекательность отдельно.
    objects["original_attractiveness"] = objects["attractiveness"]
    objects["attractiveness"] = scores.loc[objects.index].astype(float)

    if (objects["category"] == "food").sum() < meals * profile.days:
        raise RuntimeError(
            "Не хватает объектов питания для заданного числа приемов пищи "
            "(каждый объект разрешено использовать не более одного раза)"
        )

    if budget_cap == inf:
        budget_cap = float(objects["price"].sum())

    cluster_access = cluster_access_time(
        objects, objects["cluster"], hotel_lat, hotel_lon
    )
    trip = {
        "days": profile.days,
        "day_minutes": 600,  # 10:00–20:00, как в этапе 4
        "budget": budget_cap,
        "meals_per_day": meals,
        "max_per_category": 2,
    }
    best, history = plan_with_feedback(
        objects, cluster_access, hotel_lat, hotel_lon,
        trip, max_extra=feedback_extra_min,
    )
    if best is None:
        raise RuntimeError(
            "Не удалось подобрать маршрут в пределах часов работы и дня"
        )

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
    feedback_extra_min: int = 20,
    city_configs: Mapping[str, CityRouteConfig] | None = None,
) -> DecisionResult:
    """Рейтинг городов → бюджет → объекты → (по запросу) маршрут.

    Для новых городов можно передать дополнительные city_configs.
    Если каталога нет, рейтинг и смета продолжают работать.
    Проверенная продолжительность маршрута — 2 дня.
    """
    if profile.origin is None:
        raise ValueError("Укажи город отправления в TravelProfile.origin")
    for name, value in (("top_cities", top_cities), ("top_places", top_places)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} должен быть положительным целым числом")
    if (isinstance(feedback_extra_min, bool)
            or not isinstance(feedback_extra_min, int)
            or not 0 <= feedback_extra_min <= 120):
        raise ValueError("feedback_extra_min должен быть целым числом от 0 до 120")

    configs = {**CITY_ROUTE_CONFIGS, **(city_configs or {})}
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

        config = configs.get(selected_city)
        if config is None:
            notices.append(
                f"Для города {selected_city} пока нет конфигурации каталога "
                "достопримечательностей: доступны рейтинг и смета, но не маршрут."
            )
        elif not Path(config.catalog_path).is_file():
            notices.append(
                f"Для города {selected_city} пока нет каталога "
                f"{Path(config.catalog_path).name}. "
                "Добавим его позже."
            )
        else:
            all_places = recommend_places(
                selected_city,
                profile,
                top_n=None,
                catalog_path=config.catalog_path,
                city_center=config.city_center,
                max_distance_km=config.max_distance_km,
            )
            places = all_places.head(top_places).copy()

            if build_route:
                if profile.days != 2:
                    notices.append(
                        "Маршрут пока проверен только для двухдневной поездки. "
                        "Для другой продолжительности нужна проверка модели "
                        "и соответствующие сметы."
                    )
                elif not financially_allowed or estimate is None:
                    notices.append(
                        "Маршрут не строится: сначала нужна допустимая смета поездки."
                    )
                else:
                    try:
                        route = _build_city_route(
                            profile, estimate, all_places, config, feedback_extra_min
                        )
                    except (RuntimeError, ValueError) as exc:
                        notices.append(f"Маршрут не построен: {exc}")
            else:
                notices.append(
                    "Показаны рекомендации объектов, не маршрут. "
                    "Для маршрута укажи build_route=True."
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
    # Прежний тестовый сценарий Казани должен работать без изменений.
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
