import pandas as pd
import pulp
from src.optimization.objects import PROJECT_ROOT, load_objects

# Тестовые параметры одного дня поездки
TEST_DAY = {
    "day_minutes": 600,   # 10 часов, например с 10:00 до 20:00
    "budget": 3000,       # руб. на билеты и развлечения
    "transfer_min": 15,   # упрощение: переход к любому объекту - 15 мин
}


def solve_selection(objects: pd.DataFrame,
                    day_minutes: int,
                    budget: float,
                    transfer_min: int,) -> pd.DataFrame:
    """
    Модель отбора объектов на один день (задача о рюкзаке)
    x_i = 1, если объект i включен в план.
        max  Σ a_i · x_i
        s.t. Σ (t_i + τ) · x_i ≤ T   (время)
             Σ c_i · x_i ≤ B         (бюджет)
    """
    model = pulp.LpProblem("day_selection", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in objects.index}

    # Целевая функция
    model += pulp.lpSum(
        float(objects.at[i, "attractiveness"]) * x[i] for i in x), "total_attractiveness"

    # Ограничения
    model += pulp.lpSum(
        float(objects.at[i, "visit_min"] + transfer_min) * x[i] for i in x) <= day_minutes, "time_limit"
    model += pulp.lpSum(
        float(objects.at[i, "price"]) * x[i] for i in x) <= budget, "budget_limit"

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Оптимальное решение не найдено, статус: {status}")

    chosen = [i for i in x if pulp.value(x[i]) > 0.5]
    return objects.loc[chosen]


def greedy_selection(objects: pd.DataFrame,
                     day_minutes: int,
                     budget: float,
                     transfer_min: int,) -> pd.DataFrame:
    """Жадный выбор: объекты по убыванию привлекательности, пока помещаются"""
    chosen = []
    time_used = 0
    money_used = 0
    for i, row in objects.sort_values("attractiveness", ascending=False).iterrows():
        time_needed = row["visit_min"] + transfer_min
        if (time_used + time_needed <= day_minutes
                and money_used + row["price"] <= budget):
            chosen.append(i)
            time_used += time_needed
            money_used += row["price"]
    return objects.loc[chosen]


def describe(plan: pd.DataFrame, transfer_min: int) -> dict:
    """Итоговые показатели плана"""
    return {
        "Объектов": len(plan),
        "Сумма привлекательности": round(plan["attractiveness"].sum(), 2),
        "Время, мин": int((plan["visit_min"] + transfer_min).sum()),
        "Стоимость, руб.": int(plan["price"].sum()),
    }


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    objects = load_objects(
        PROJECT_ROOT / "data" / "raw" / "kazan_objects.csv"
    ).set_index("id")
    params = TEST_DAY

    optimal = solve_selection(objects, **params)
    greedy = greedy_selection(objects, **params)

    columns = ["name", "visit_min", "price", "attractiveness"]
    print("Оптимальный план на день (PuLP, CBC):")
    print(optimal.sort_values("attractiveness", ascending=False)[columns]
          .round(2).to_string())
    print()
    print("Жадный план (по убыванию привлекательности):")
    print(greedy[columns].round(2).to_string())
    print()

    optimal_stats = describe(optimal, params["transfer_min"])
    greedy_stats = describe(greedy, params["transfer_min"])
    print(f"{'Показатель':<26}{'Оптимизация':>14}{'Жадный':>10}")
    for key in optimal_stats:
        print(f"{key:<26}{optimal_stats[key]:>14}{greedy_stats[key]:>10}")
