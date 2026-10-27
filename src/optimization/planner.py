import pandas as pd
from src.optimization.objects import PROJECT_ROOT, load_objects
from src.optimization.routing import DAY_END, solve_day_route
from src.optimization.travel import (
    access_time,
    cluster_access_time,
    cluster_objects,
)
from src.optimization.trip_plan import (
    BASE_ID,
    TEST_TRIP,
    add_cluster_travel,
    solve_clustered_trip,
    summarize_days,
)

BUDGET_STEP_MIN = 10     # шаг увеличения бюджета времени модели отбора
MAX_EXTRA_MIN = 120      # максимальная надбавка к бюджету


def clock_to_minutes(value: str) -> int:
    """'ЧЧ:ММ' → минуты от полуночи"""
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def route_all_days(plan, hotel_lat, hotel_lon):
    """
    Строит маршрут для каждого дня плана
    Возвращает словарь {день: расписание} или None,
    если хотя бы один день не укладывается в часы работы и 20:00
    """
    routes = {}
    for day, group in plan.groupby("day"):
        try:
            routes[day] = solve_day_route(group, hotel_lat, hotel_lon)
        except RuntimeError:
            return None
    return routes


def plan_with_feedback(
    objects: pd.DataFrame,
    cluster_access: pd.Series,
    hotel_lat: float,
    hotel_lon: float,
    trip: dict,
    step: int = BUDGET_STEP_MIN,
    max_extra: int = MAX_EXTRA_MIN,):
    """
    Обратная связь между уровнями модели
    Модель отбора (верхний уровень) оценивает дорогу с запасом,
    поэтому ей постепенно увеличивают бюджет времени дня
    Каждый план проверяется моделью маршрута (нижний уровень)
    Поиск останавливается на первом плане, который нельзя выполнить
    до 20:00, и возвращает последний выполнимый план
    """
    history = []
    best = None
    base_budget = trip["day_minutes"]
    for budget in range(base_budget, base_budget + max_extra + 1, step):
        plan = solve_clustered_trip(
            objects, cluster_access, **{**trip, "day_minutes": budget}
        )
        routes = route_all_days(plan, hotel_lat, hotel_lon)
        feasible = routes is not None

        returns = (
            {day: r["Прибытие"].iloc[-1] for day, r in routes.items()}
            if feasible else {}
        )
        history.append({
            "Бюджет модели, мин": budget,
            "Объектов": len(plan),
            "Привлекательность": round(plan["attractiveness"].sum(), 2),
            "Маршрут выполним": "да" if feasible else "нет",
            "Возвращение по дням": ", ".join(returns.values()),
        })
        if not feasible:
            break
        best = (budget, plan, routes)
    return best, pd.DataFrame(history)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    objects = load_objects(PROJECT_ROOT / "data" / "raw" / "kazan_objects.csv").set_index("id")
    base = objects.loc[BASE_ID]
    objects["access_min"] = access_time(objects, base["lat"], base["lon"])
    objects["cluster"] = cluster_objects(objects)
    cluster_access = cluster_access_time(objects, objects["cluster"], base["lat"], base["lon"])

    best, history = plan_with_feedback(objects, cluster_access, base["lat"], base["lon"], TEST_TRIP)
    print("Итерации обратной связи:")
    print(history.to_string(index=False))
    print()

    budget, plan, routes = best
    first = history.iloc[0]["Привлекательность"]
    final = plan["attractiveness"].sum()
    print(f"Выбран бюджет модели отбора: {budget} мин")
    print(f"Привлекательность: {first:.2f} → {final:.2f} "
          f"(+{(final / first - 1) * 100:.1f}%)")
    print(f"Стоимость: {plan['price'].sum()} из {TEST_TRIP['budget']} руб.")
    print()

    for day, route in routes.items():
        spare = DAY_END - clock_to_minutes(route["Прибытие"].iloc[-1])
        print(f"=== День {day}: расписание ===")
        print(route.to_string(index=False))
        print(f"Время в пути: {route['В пути, мин'].sum()} мин; "
              f"возвращение: {route['Прибытие'].iloc[-1]}; "
              f"запас до 20:00: {spare} мин")
        print()
