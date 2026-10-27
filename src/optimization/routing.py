import pandas as pd
import pulp
from src.optimization.objects import PROJECT_ROOT, load_objects
from src.optimization.travel import (
    access_time,
    cluster_access_time,
    cluster_objects,
    travel_time_matrix,
)
from src.optimization.trip_plan import BASE_ID, TEST_TRIP, solve_clustered_trip

HOTEL = "hotel"
DAY_START = 10 * 60   # 10:00, минуты от полуночи
DAY_END = 20 * 60     # 20:00
LUNCH_WINDOW = (12 * 60, 15 * 60)  # начало посещения объектов food: 12:00-15:00
RETURN_TIME_WEIGHT = 0.001        # малый вес: при равной дороге - вернуться раньше
ROUTE_TIME_LIMIT = 30              # секунд на поиск маршрута одного дня


def to_clock(minutes: float) -> str:
    """Минуты от полуночи → 'ЧЧ:ММ'"""
    minutes = int(round(minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def solve_day_route(
    day_objects: pd.DataFrame,
    hotel_lat: float,
    hotel_lon: float,
    day_start: int = DAY_START,
    day_end: int = DAY_END,
    lunch_window: tuple[int, int] | None = LUNCH_WINDOW,
    time_limit: int = ROUTE_TIME_LIMIT,) -> pd.DataFrame:
    """
    Порядок посещения объектов одного дня - задача коммивояжера
    с временными окнами (TSPTW). Маршрут начинается и заканчивается
    в месте проживания.
    z[i, j] = 1, если после точки i идем в точку j;
    s[i] - время начала посещения точки i (минуты от полуночи)
        min  Σ t_ij · z[i, j]                      суммарное время в пути
        s.t. в каждую точку входим и выходим ровно один раз
             s_j ≥ s_i + v_i + t_ij − M(1 − z[i, j])   время согласовано
             open_j ≤ s_j,  s_j + v_j ≤ close_j      часы работы
             возвращение в гостиницу не позже day_end
             объекты food начинаются внутри окна обеда
    """
    hotel = pd.DataFrame(
        {"lat": [hotel_lat], "lon": [hotel_lon], "visit_min": [0],
         "open_min": [day_start], "close_min": [day_end]},
        index=[HOTEL],
    )
    points = pd.concat([hotel, day_objects[hotel.columns]])
    travel = travel_time_matrix(points)
    nodes = list(points.index)
    objects_only = nodes[1:]
    big_m = 2 * 24 * 60

    model = pulp.LpProblem("day_route", pulp.LpMinimize)
    z = {
        (i, j): pulp.LpVariable(f"z_{i}_{j}", cat="Binary")
        for i in nodes for j in nodes if i != j
    }
    s = {i: pulp.LpVariable(f"s_{i}", lowBound=0) for i in nodes}
    end = pulp.LpVariable("return_time", lowBound=0)

    model += (
            pulp.lpSum(float(travel.at[i, j]) * z[i, j] for (i, j) in z)
            + RETURN_TIME_WEIGHT * end), "total_travel"
    for j in nodes:
        model += pulp.lpSum(z[i, j] for i in nodes if i != j) == 1, f"in_{j}"
        model += pulp.lpSum(z[j, k] for k in nodes if k != j) == 1, f"out_{j}"
    model += s[HOTEL] == day_start, "start_time"

    for (i, j) in z:
        arrival = s[i] + float(points.at[i, "visit_min"] + travel.at[i, j])
        if j == HOTEL:
            model += end >= arrival - big_m * (1 - z[i, j]), f"back_{i}"
        else:
            model += s[j] >= arrival - big_m * (1 - z[i, j]), f"time_{i}_{j}"

    for j in objects_only:
        model += s[j] >= float(points.at[j, "open_min"]), f"open_{j}"
        model += (s[j] + float(points.at[j, "visit_min"])
                  <= float(points.at[j, "close_min"])), f"close_{j}"
    model += end <= day_end, "day_end"
    if lunch_window is not None:
        lunch_start, lunch_end = lunch_window
        food = day_objects.index[day_objects["category"] == "food"]
        for j in food:
            model += s[j] >= lunch_start, f"lunch_from_{j}"
            model += s[j] <= lunch_end, f"lunch_to_{j}"
    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(
            f"Маршрут не построен (статус: {status}). Вероятно, объекты дня "
            "не помещаются в часы работы или в продолжительность дня"
        )

    # Восстанавливаем порядок обхода, начиная из гостиницы
    next_point = {i: j for (i, j) in z if pulp.value(z[i, j]) > 0.5}
    rows = []
    current = HOTEL
    clock = float(day_start)
    while True:
        following = next_point[current]
        leave = (clock if current == HOTEL
                 else pulp.value(s[current]) + points.at[current, "visit_min"])
        arrive = leave + travel.at[current, following]
        if following == HOTEL:
            rows.append({"Точка": "Возвращение в гостиницу",
                         "В пути, мин": int(travel.at[current, following]),
                         "Прибытие": to_clock(arrive), "Начало": "",
                         "Ожидание, мин": 0, "Уход": ""})
            break
        start = pulp.value(s[following])
        rows.append({
            "Точка": day_objects.at[following, "name"],
            "В пути, мин": int(travel.at[current, following]),
            "Прибытие": to_clock(arrive),
            "Начало": to_clock(start),
            "Ожидание, мин": int(round(start - arrive)),
            "Уход": to_clock(start + points.at[following, "visit_min"]),
        })
        current, clock = following, start
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    objects = load_objects(PROJECT_ROOT / "data" / "raw" / "kazan_objects.csv").set_index("id")
    base = objects.loc[BASE_ID]
    objects["access_min"] = access_time(objects, base["lat"], base["lon"])
    objects["cluster"] = cluster_objects(objects)
    cluster_access = cluster_access_time(objects, objects["cluster"], base["lat"], base["lon"])
    plan = solve_clustered_trip(objects, cluster_access, **TEST_TRIP)

    for day, group in plan.groupby("day"):
        route = solve_day_route(group, base["lat"], base["lon"])
        total_travel = route["В пути, мин"].sum()
        print(f"=== День {day}: расписание ===")
        print(route.to_string(index=False))
        print(f"Время в пути: {total_travel} мин; "
              f"возвращение: {route['Прибытие'].iloc[-1]}")
        print()
