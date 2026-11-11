import pandas as pd
from time import monotonic
from src.optimization.objects import PROJECT_ROOT, load_objects
from src.optimization.routing import DAY_END, ROUTE_TIME_LIMIT, solve_day_route
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


def route_all_days(plan, hotel_lat, hotel_lon, *, deadline: float | None = None):
    """Маршруты всех дней. Deadline используется только в интерфейсном режиме."""
    routes = {}
    for day, group in plan.groupby("day"):
        # Не начинаем еще одну задачу, если общий лимит уже закончился.
        time_limit = ROUTE_TIME_LIMIT
        if deadline is not None:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Истек общий лимит времени планирования")
            time_limit = min(float(ROUTE_TIME_LIMIT), max(0.1, remaining))
        try:
            routes[day] = solve_day_route(
                group, hotel_lat, hotel_lon, time_limit=time_limit
            )
        except RuntimeError:
            # Набор объектов может быть несовместим с временными окнами.
            # В том числе решатель может не успеть найти маршрут за лимит.
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError("Истек общий лимит времени планирования")
            return None
    return routes


def plan_with_feedback(
    objects: pd.DataFrame,
    cluster_access: pd.Series,
    hotel_lat: float,
    hotel_lon: float,
    trip: dict,
    step: int = BUDGET_STEP_MIN,
    max_extra: int = MAX_EXTRA_MIN,
    max_seconds: float | None = None,
):
    """Согласование отбора объектов и расписаний.

    max_seconds=None сохраняет исходное поведение этапа 4 и тестов.
    В интерфейсе общий дедлайн ограничивает все повторные вызовы CBC.
    Если один допустимый план уже построен, возвращается он, даже когда
    более сложная следующая итерация не уложилась в лимит времени.
    """
    if max_seconds is not None and max_seconds <= 0:
        raise TimeoutError("Время планирования истекло")
    deadline = monotonic() + max_seconds if max_seconds is not None else None

    history = []
    best = None
    base_budget = trip["day_minutes"]
    for budget in range(base_budget, base_budget + max_extra + 1, step):
        remaining = deadline - monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            if best is None:
                raise TimeoutError("Не удалось получить маршрут за отведенное время")
            break

        # Оставляем время и для маршрутизации по дням: CBC верхнего
        # уровня получает не более 45% остатка и максимум 20 секунд.
        upper_limit = (
            None if remaining is None
            else max(0.1, min(20.0, remaining * 0.45))
        )
        try:
            plan = solve_clustered_trip(
                objects, cluster_access,
                **{**trip, "day_minutes": budget},
                solver_time_limit=upper_limit,
            )
        except TimeoutError:
            if best is None:
                raise
            break

        try:
            routes = route_all_days(
                plan, hotel_lat, hotel_lon, deadline=deadline
            )
        except TimeoutError:
            if best is None:
                raise
            break
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
