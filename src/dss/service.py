"""Этап 8. Единая точка входа для выбора города и достопримечательностей
Здесь объединены модули этапа 7 (AHP/риск, бюджет, база знаний)
"""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from src.multicriteria.alternatives import PROJECT_ROOT
from src.recommendations.budget import BudgetLimits, calculate_budget
from src.recommendations.city_budget import recommend_cities_with_budget
from src.recommendations.cost_catalog import load_cost_estimates
from src.recommendations.places import recommend_places
from src.recommendations.profile import TravelProfile


DEFAULT_COSTS_PATH = PROJECT_ROOT / "data" / "test" / "city_costs.csv"


@dataclass
class DecisionResult:
    """Результат первого шага СППР для будущего интерфейса"""
    all_cities: pd.DataFrame
    affordable_cities: pd.DataFrame
    selected_city: str | None
    city_budget: dict | None
    places: pd.DataFrame
    notices: list[str]


def analyze_trip(
    profile: TravelProfile,
    selected_city: str | None = None,
    top_cities: int = 5,
    top_places: int = 10,
    costs_path: str | Path = DEFAULT_COSTS_PATH,) -> DecisionResult:
    """Объединяет рекомендации городов, расчет бюджета и подбор объектов"""
    if profile.origin is None:
        raise ValueError("Укажи город отправления в TravelProfile.origin")
    for name, value in (
        ("top_cities", top_cities),
        ("top_places", top_places),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} должен быть положительным целым числом")
    # 1. Загружаем сметы поездок
    estimates = load_cost_estimates(costs_path)
    # 2. Получаем рейтинг всех городов с проверкой бюджета
    all_cities = recommend_cities_with_budget(
        profile,
        estimates,
        top_n=None,
        only_affordable=False,
    )
    # 3. Выбираем лучшие финансово допустимые города
    affordable = (
        all_cities.loc[all_cities["budget_ok"].eq(True)]
        .head(top_cities)
        .copy()
        .reset_index(drop=True)
    )
    notices = [
        "Цены и тематические оценки в data/test/ учебные; "
        "они не являются актуальными предложениями перевозчиков и гостиниц"
    ]
    if all_cities["budget_ok"].isna().all():
        notices.append(
            f"Нет смет для отправления из {profile.origin} "
            f"на {profile.days} дн.: финансовую доступность "
            "городов определить нельзя."
        )
    elif affordable.empty:
        notices.append(
            "Нет направлений, проходящих все указанные бюджетные лимиты"
        )
    budget = None
    places = pd.DataFrame()
    # 4. Если пользователь выбрал направление,
    # получаем подробную смету и рекомендации объектов
    if selected_city is not None:
        if selected_city not in set(all_cities["city"]):
            raise ValueError(
                f"Неизвестный город назначения: {selected_city}"
            )
        selected_row = all_cities.set_index("city").loc[selected_city]
        estimate = next(
            (
                x for x in estimates
                if x.origin == profile.origin
                and x.city == selected_city
                and x.days == profile.days
            ),
            None,
        )
        # 5. Рассчитываем подробную смету выбранного города
        if estimate is not None:
            budget = calculate_budget(
                estimate.costs,
                BudgetLimits.from_profile(profile),
            )
            budget["source"] = estimate.source
            budget["checked_on"] = estimate.checked_on
        if selected_row["budget_status"] != "в пределах лимитов":
            notices.append(
                f"{selected_city}: {selected_row['budget_status']}. "
                "Это направление пока не подтверждено "
                "как финансово допустимое."
            )
        # 6. Рекомендуем объекты выбранного города
        if selected_city == "Казань":
            places = recommend_places(
                "Казань",
                profile,
                top_n=top_places,
            )
            notices.append(
                "Список объектов — рейтинг, а не готовый маршрут. "
                "Их общую стоимость, расписание и переезды проверим "
                "при подключении оптимизатора на следующем шаге."
            )
        else:
            notices.append(
                f"Для города {selected_city} пока нет каталога объектов: "
                "планирование маршрута доступно только для Казани."
            )
    # 7. Возвращаем единый результат для интерфейса
    return DecisionResult(
        all_cities=all_cities,
        affordable_cities=affordable,
        selected_city=selected_city,
        city_budget=budget,
        places=places,
        notices=notices,
    )


if __name__ == "__main__":

    profile = TravelProfile(
        origin="Москва",
        days=2,
        total_budget=30_000,
        tickets_budget=15_000,
        activity_budget=3_000,
        interests=("Культура и история",),
        risk_attitude="нейтральный",
    )

    result = analyze_trip(
        profile,
        selected_city="Казань",
    )

    print("Топ финансово допустимых городов:")
    print(
        result.affordable_cities[
            ["city", "score", "estimated_total"]
        ].round(2).to_string(index=False)
    )

    print("\nВыбранный город:", result.selected_city)

    if result.city_budget:
        print(
            "Смета:",
            f"{result.city_budget['total']:,.0f} руб."
        )
        print(
            "Бюджет соблюден:",
            result.city_budget["fits_limits"]
        )

    print("\nТоп-5 достопримечательностей:")
    print(
        result.places[
            ["name", "score", "price"]
        ].head(5).round(2).to_string(index=False)
    )

    print("\nПримечания:")
    for notice in result.notices:
        print("-", notice)
