import pandas as pd
import pulp
from src.optimization.objects import PROJECT_ROOT, load_objects
from src.optimization.travel import access_time

# Тестовые параметры поездки
TEST_TRIP = {
    "days": 2,
    "day_minutes": 600,        # 10 часов в день
    "budget": 3000,            # руб. на билеты и развлечения за всю поездку
    "meals_per_day": 1,        # минимум объектов категории food в день
    "max_per_category": 2,     # максимум объектов одной категории в день
}
BASE_ID = "kremlin"            # место проживания - рядом с Кремлем


def solve_trip(
    objects: pd.DataFrame,
    days: int,
    day_minutes: int,
    budget: float,
    meals_per_day: int = 0,
    max_per_category: int | None = None,) -> pd.DataFrame:
    """
    Модель распределения объектов по дням поездки
    x[i, d] = 1, если объект i посещается в день d
        max  Σ_i Σ_d a_i · x[i, d]
        s.t. Σ_d x[i, d] ≤ 1                         каждый объект не более раза
             Σ_i (t_i + r_i) · x[i, d] ≤ T          время в каждый день
             Σ_i Σ_d c_i · x[i, d] ≤ B               бюджет поездки
             Σ_{i ∈ food} x[i, d] ≥ m               прием пищи в каждый день
             Σ_{i ∈ k} x[i, d] ≤ K                  разнообразие категорий
    t_i - посещение, r_i - дорога (столбец access_min)
    """
    day_list = range(1, days + 1)
    model = pulp.LpProblem("trip_plan", pulp.LpMaximize)

    x = {
        (i, d): pulp.LpVariable(f"x_{i}_day{d}", cat="Binary")
        for i in objects.index for d in day_list}

    def value(i, column):
        return float(objects.at[i, column])

    # Целевая функция
    model += pulp.lpSum(value(i, "attractiveness") * x[i, d] for (i, d) in x), "total_attractiveness"

    # Каждый объект - не более одного раза за поездку
    for i in objects.index:
        model += pulp.lpSum(x[i, d] for d in day_list) <= 1, f"once_{i}"

    for d in day_list:
        # Время в день: посещение + дорога
        model += pulp.lpSum(
            (value(i, "visit_min") + value(i, "access_min")) * x[i, d]
            for i in objects.index) <= day_minutes, f"time_day{d}"

        # Прием пищи
        food = objects.index[objects["category"] == "food"]
        model += pulp.lpSum(x[i, d] for i in food) >= meals_per_day, f"meal_day{d}"

        # Разнообразие: не больше K объектов одной категории в день
        if max_per_category is not None:
            for category in objects["category"].unique():
                members = objects.index[objects["category"] == category]
                model += pulp.lpSum(x[i, d] for i in members) <= max_per_category, f"cat_{category}_day{d}"

    # Бюджет поездки
    model += pulp.lpSum(value(i, "price") * x[i, d] for (i, d) in x) <= budget, "budget"
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Оптимальное решение не найдено, статус: {status}")

    chosen = [(i, d) for (i, d) in x if pulp.value(x[i, d]) > 0.5]
    plan = objects.loc[[i for i, _ in chosen]].copy()
    plan.insert(0, "day", [d for _, d in chosen])
    return plan.sort_values(["day", "attractiveness"], ascending=[True, False])


def summarize_days(plan: pd.DataFrame, day_minutes: int) -> pd.DataFrame:
    """Показатели плана по дням"""
    rows = []
    for day, group in plan.groupby("day"):
        busy = (group["visit_min"] + group["access_min"]).sum()
        rows.append({
            "День": day,
            "Объектов": len(group),
            "Привлекательность": round(group["attractiveness"].sum(), 2),
            "Посещение, мин": int(group["visit_min"].sum()),
            "Дорога, мин": int(group["access_min"].sum()),
            "Свободно, мин": int(day_minutes - busy),
            "Стоимость, руб.": int(group["price"].sum()),
        })
    return pd.DataFrame(rows).set_index("День")


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    objects = load_objects(
        PROJECT_ROOT / "data" / "raw" / "kazan_objects.csv"
    ).set_index("id")
    base = objects.loc[BASE_ID]
    objects["access_min"] = access_time(objects, base["lat"], base["lon"])

    print("Время на дорогу к удаленным объектам (туда и обратно), мин:")
    print(objects.nlargest(5, "access_min")[["name", "access_min"]].to_string())
    print()

    variants = {
        "Базовая модель (время и бюджет)": {
            **TEST_TRIP, "meals_per_day": 0, "max_per_category": None,
        },
        "Уточненная модель (+ обед, + разнообразие)": TEST_TRIP,
    }
    columns = ["day", "name", "category", "visit_min", "access_min", "price", "attractiveness"]

    for title, params in variants.items():
        plan = solve_trip(objects, **params)
        print(f"=== {title} ===")
        print(plan[columns].round(2).to_string())
        print()
        print(summarize_days(plan, params["day_minutes"]).to_string())
        print(f"Итого: привлекательность {plan['attractiveness'].sum():.2f}, "
              f"стоимость {plan['price'].sum()} из {params['budget']} руб.")
        print()
