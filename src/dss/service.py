"""Этап 8. Единый сервис выбора города, бюджета и персонального маршрута
Маршруты строятся для городов с каталогом объектов в формате этапа 4
Поддерживаемая продолжительность: 1–3 дня, время 10:00–20:00.
"""

from dataclasses import dataclass, replace
from math import inf, isfinite
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.forecast.explore import monthly_series
from src.forecast.forecast import (
    FORECAST_YEARS, fit_temperature, predict_temperature, rain_scenarios,
)
from src.forecast.model import to_table
from src.forecast.weather import load_weather
from src.knowledge.facts import TOURIST, add_profile
from src.knowledge.inference import forward_chain
from src.knowledge.kb import KnowledgeBase
from src.knowledge.rules import RULES
from src.multicriteria.alternatives import (
    CRITERIA_COLUMNS, PROJECT_ROOT, load_alternatives,
)
from src.optimization.objects import load_objects
from src.optimization.planner import plan_with_feedback
from src.optimization.travel import cluster_access_time, cluster_objects
from src.optimization.trip_plan import BASE_ID
from src.recommendations.budget import BudgetLimits, calculate_budget
from src.recommendations.city_budget import CityCostEstimate, recommend_cities_with_budget
from src.recommendations.cost_catalog import (
    load_cost_estimates, expand_test_cost_estimates,
)
from src.recommendations.cities import (
    DESTINATIONS_PATH, UNCERTAINTY_PATH, load_city_themes,
)
from src.recommendations.places import KAZAN_CATALOG, recommend_places
from src.recommendations.profile import THEMES, TravelProfile
from src.uncertainty.scenarios import (
    WEATHER_STATES, build_payoff_matrix, build_scenarios, load_uncertainty,
)


DEFAULT_COSTS_PATH = PROJECT_ROOT / "data" / "test" / "city_costs.csv"


@dataclass(frozen=True)
class CityRouteConfig:
    """Настройки маршрута для города с каталогом достопримечательностей
    city_center - центр проверки координат (широта, долгота)
    start_coordinates - условное место проживания
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


# Каталоги новых городов находятся в data/raw/. Центры городов
# используются как условные места проживания
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
        city_center=(54.706129, 20.511063),
        start_coordinates=(54.706129, 20.511063),
    ),
    "Сочи": CityRouteConfig(
        catalog_path=PROJECT_ROOT / "data" / "raw" / "sochi_objects.csv",
        city_center=(43.5855, 39.7231),
        start_coordinates=(43.5855, 39.7231),
        max_distance_km=60.0,
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


@dataclass(frozen=True)
class WeatherResult:
    """Погодная информация для Казани из модели этапа 6
    Температура - ПРОГНОЗ СРЕДНЕГО ЗА МЕСЯЦ
    Вероятности - частоты исторических окон 2011-2025
    Байесовские оценки - анализ чувствительности только Казани
    """
    city: str
    year: int
    month: int
    days: int
    temperature_c: float
    lower_80_c: float
    upper_80_c: float
    p_better: float
    p_normal: float
    p_worse: float
    historical_windows: int
    bayes_expert: float
    bayes_weather_data: float


@dataclass
class DecisionResult:
    all_cities: pd.DataFrame
    affordable_cities: pd.DataFrame
    selected_city: str | None
    city_budget: dict | None
    places: pd.DataFrame
    notices: list[str]
    route: RouteResult | None = None
    weather: WeatherResult | None = None


def _kazan_weather_report(profile: TravelProfile, year: int, month: int,) -> WeatherResult:
    """Температура, осадки и чувствительность Байеса для Казани
    Методики берутся из этапов 3 и 6; новые модели не обучаются
    Соответствие сценариев этапа 3 и этапа 6 - допущение:
    без дождя = «лучше», дождь в части дней = «обычная»,
    дождь все дни = «хуже»
    """
    weather = load_weather()                      # ERA5 1991–2025
    monthly = to_table(monthly_series(weather))
    temperature_model = fit_temperature(monthly)
    predicted = predict_temperature(
        temperature_model, [year], [month]
    ).iloc[0]
    probabilities = rain_scenarios(
        weather, trip_days=profile.days
    ).set_index("month").loc[month]

    # Повторяем оценки города из этапа 7: тематическая персонализация,
    # критерии AHP и чувствительность 9 сценариев этапа 3
    cities, scores = load_alternatives(DESTINATIONS_PATH)
    index = cities.index("Казань")
    city_scores = scores[index:index + 1].copy()
    if profile.interests:
        themes = load_city_themes()
        selected_themes = [THEMES[name] for name in profile.interests]
        interest_score = float(themes.loc["Казань", selected_themes].mean())
    else:
        interest_score = 3.0
    city_scores[0, CRITERIA_COLUMNS.index("interests")] = interest_score
    scenarios = build_scenarios()
    uncertainty = load_uncertainty(UNCERTAINTY_PATH, ["Казань"])
    payoff = build_payoff_matrix(
        ["Казань"], city_scores, uncertainty, scenarios,
        profile.criterion_weights(),
    ).loc["Казань"].to_numpy(dtype=float)

    # Для новой оценки Байеса сохраняем экспертные вероятности цен,
    # заменяя только вероятности трех погодных сценариев
    base_weather_prob = {name: probability for name, _, probability in WEATHER_STATES}
    revised_weather_prob = {
        "погода лучше": float(probabilities["p_better"]),
        "погода обычная": float(probabilities["p_normal"]),
        "погода хуже": float(probabilities["p_worse"]),
    }
    expert_prob = scenarios["probability"].to_numpy(dtype=float)
    updated_prob = np.array([
        float(row["probability"])
        * revised_weather_prob[row["weather_state"]]
        / base_weather_prob[row["weather_state"]]
        for _, row in scenarios.iterrows()
    ])
    if not np.isclose(updated_prob.sum(), 1.0):
        raise ValueError("Вероятности погодных сценариев не суммируются в 1")

    return WeatherResult(
        city="Казань", year=year, month=month, days=profile.days,
        temperature_c=float(predicted["t_forecast"]),
        lower_80_c=float(predicted["t_low"]),
        upper_80_c=float(predicted["t_high"]),
        p_better=revised_weather_prob["погода лучше"],
        p_normal=revised_weather_prob["погода обычная"],
        p_worse=revised_weather_prob["погода хуже"],
        historical_windows=int(probabilities["windows"]),
        bayes_expert=float(np.dot(payoff, expert_prob)),
        bayes_weather_data=float(np.dot(payoff, updated_prob)),
    )


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


def _start_point(config: CityRouteConfig, objects: pd.DataFrame) -> tuple[float, float]:
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

    # Как в старой версии: кластеры вычисляются на всем каталоге до фильтров
    objects["cluster"] = cluster_objects(objects)
    scores = ranked_places.set_index("object_id")["score"]
    objects = objects.loc[objects.index.intersection(scores.index)].copy()
    if objects.empty:
        raise RuntimeError("После фильтрации не осталось доступных объектов")

    # Оптимизатор этапа 4 максимизирует столбец attractiveness. Подставляем
    # в него персональные оценки, сохранив исходную привлекательность отдельно
    objects["original_attractiveness"] = objects["attractiveness"]
    objects["attractiveness"] = scores.loc[objects.index].astype(float)

    available_food = int((objects["category"] == "food").sum())
    if available_food < meals:
        raise RuntimeError("В каталоге недостаточно кафе даже для одного дня")
    # При нехватке разных кафе разрешаем повторить кафе в другой день
    # Другие достопримечательности повторять нельзя
    repeat_food = available_food < meals * profile.days

    if budget_cap == inf:
        budget_cap = float(objects["price"].sum())

    cluster_access = cluster_access_time(
        objects, objects["cluster"], hotel_lat, hotel_lon
    )
    trip = {
        "days": profile.days,
        # При двух приемах пищи оставляем 120 мин резерва на разделенные
        # окна обеда и ужина: фактический день всё равно 10:00–20:00
        "day_minutes": 480 if meals == 2 else 600,
        "budget": budget_cap,
        "meals_per_day": meals,
        "max_per_category": 2,
        "allow_repeat_food": repeat_food,
    }
    # При обеде и ужине два разнесенных временных окна оставляют меньше
    # свободы для посещений. Если жадная по полезности верхняя модель
    # выбрала набор, не помещающийся в расписание, уменьшаем ее временной
    # лимит и ищем более компактный маршрут. Реальный день 10–20 не меняем
    daily_limits = (480, 420, 360) if meals == 2 else (600,)
    best = None
    history = pd.DataFrame()
    for candidate_minutes in daily_limits:
        trip["day_minutes"] = candidate_minutes
        best, history = plan_with_feedback(
            objects, cluster_access, hotel_lat, hotel_lon,
            trip, max_extra=feedback_extra_min,
        )
        if best is not None:
            break
    if best is None:
        raise RuntimeError(
            "Не удалось подобрать маршрут в пределах часов работы и дня "
            "(проверены более компактные планы с обедом и ужином)"
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
    travel_year: int | None = None,
    travel_month: int | None = None,
) -> DecisionResult:
    """Рейтинг городов → бюджет → объекты → (по запросу) маршрут.
    Для новых городов можно передать дополнительные city_configs.
    Если каталога нет, рейтинг и смета продолжают работать.
    Поддерживаемая продолжительность маршрута - от 1 до 3 дней
    """
    if profile.origin is None:
        raise ValueError("Укажи город отправления в TravelProfile.origin")
    if profile.days not in (1, 2, 3):
        raise ValueError("На этапе 8 поддерживаются поездки на 1, 2 или 3 дня")
    if (travel_year is None) != (travel_month is None):
        raise ValueError("Для прогноза укажи и travel_year, и travel_month")
    if travel_year is not None:
        if (isinstance(travel_year, bool) or not isinstance(travel_year, int)
                or travel_year not in FORECAST_YEARS):
            raise ValueError(f"Прогноз этапа 6 доступен только на годы {FORECAST_YEARS}")
        if (isinstance(travel_month, bool) or not isinstance(travel_month, int)
                or not 1 <= travel_month <= 12):
            raise ValueError("travel_month должен быть целым числом от 1 до 12")
    for name, value in (("top_cities", top_cities), ("top_places", top_places)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} должен быть положительным целым числом")
    if (isinstance(feedback_extra_min, bool)
            or not isinstance(feedback_extra_min, int)
            or not 0 <= feedback_extra_min <= 120):
        raise ValueError("feedback_extra_min должен быть целым числом от 0 до 120")

    configs = {**CITY_ROUTE_CONFIGS, **(city_configs or {})}
    estimates = expand_test_cost_estimates(load_cost_estimates(costs_path))
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
    if profile.days != 2 and any(
        e.days == profile.days and "УЧЕБНЫЙ пересчет" in e.source
        for e in estimates
    ):
        notices.append(
            "Сметы на 1 и 3 дня пересчитаны из учебных двухдневных: "
            "длительность поездки не учитывает время дороги до города."
        )
    if all_cities["budget_ok"].isna().all():
        notices.append(
            f"Нет смет для отправления из {profile.origin} на {profile.days} дн."
        )
    elif affordable.empty:
        notices.append("Нет направлений, проходящих все бюджетные лимиты.")

    budget = None
    places = pd.DataFrame()
    route = None
    weather_result = None

    if travel_year is not None and selected_city is None:
        notices.append("Чтобы показать погодную информацию, сначала выбери город.")

    if selected_city is not None:
        if selected_city not in set(all_cities["city"]):
            raise ValueError(f"Неизвестный город назначения: {selected_city}")
        if travel_year is not None:
            if selected_city == "Казань":
                try:
                    weather_result = _kazan_weather_report(
                        profile, travel_year, travel_month
                    )
                    notices.append(
                        "Этап 6: температура - прогноз среднего за месяц, "
                        "погодные вероятности - исторические частоты 2011–2025. "
                        "Казань пересчитана по Байесу отдельно; "
                        "общий рейтинг городов не изменен. "
                        "Погодный сценарий маршрута задается вручную."
                    )
                except (FileNotFoundError, ValueError) as exc:
                    notices.append(f"Погодная модель Казани недоступна: {exc}")
            else:
                notices.append(
                    f"Для города {selected_city} нет модели этапа 6: "
                    "прогноз Казани не переносится на другие направления."
                )
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
                "Добавим его на следующем шаге."
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
                if estimate is None:
                    notices.append("Маршрут не строится: отсутствует смета поездки.")
                elif (profile.tickets_budget is not None and
                      estimate.costs.round_trip_tickets > profile.tickets_budget):
                    notices.append("Маршрут не строится: превышен лимит билетов.")
                elif (profile.total_budget is not None and
                      (estimate.costs.round_trip_tickets
                       + estimate.costs.price_per_night * estimate.costs.nights
                       + estimate.costs.local_transport)
                      * (1 + estimate.costs.reserve_fraction) > profile.total_budget):
                    notices.append(
                        "Маршрут не строится: обязательные расходы уже превышают бюджет."
                    )
                else:
                    # Предварительная смета может быть завышена: оптимизатор
                    # иногда находит набор объектов дешевле плановой оценки
                    try:
                        route = _build_city_route(
                            profile, estimate, all_places, config, feedback_extra_min
                        )
                        if not financially_allowed:
                            notices.append(
                                "После оптимизации фактические расходы на объекты "
                                "снизились: итоговый бюджет маршрута соблюден."
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
        weather=weather_result,
    )


if __name__ == "__main__":
    profile = TravelProfile(
        origin="Москва", days=2, total_budget=30_000,
        tickets_budget=15_000, activity_budget=3_000,
        interests=("Культура и история",), risk_attitude="нейтральный",
    )
    result = analyze_trip(
        profile, selected_city="Казань", build_route=True,
        travel_year=2027, travel_month=7,
    )
    print("Топ финансово допустимых городов:")
    print(result.affordable_cities[
        ["city", "score", "estimated_total"]
    ].round(2).to_string(index=False))
    print("\nКазань: предварительная смета:",
          f"{result.city_budget['total']:,.0f} руб.")
    if result.weather is not None:
        w = result.weather
        print(f"\nПогода Казани на {w.month:02d}.{w.year} ({w.days} дн.):")
        print(f"Температура месяца: {w.temperature_c:.1f} °C "
              f"(80%-й интервал {w.lower_80_c:.1f}–{w.upper_80_c:.1f} °C)")
        print(f"Вероятности: сухо {w.p_better:.3f}; "
              f"дождь в части дней {w.p_normal:.3f}; "
              f"дождь во все дни {w.p_worse:.3f}")
        print(f"Байес Казани: эксперты {w.bayes_expert:.3f}; "
              f"данные этапа 6 {w.bayes_weather_data:.3f}")
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
