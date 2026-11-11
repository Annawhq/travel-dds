"""Проверка смет и маршрутов Travel DSS на 1–3 дня.

Из корня репозитория:
    python tests/check_trip_durations.py

Планирование требует PuLP/CBC и каталоги 4 городов в data/raw/.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dss.service import analyze_trip
from src.recommendations.cost_catalog import (load_cost_estimates, expand_test_cost_estimates)
from src.recommendations.profile import TravelProfile


CITIES = ("Казань", "Нижний Новгород", "Калининград", "Сочи")


def minutes(value: str) -> int:
    hour, minute = map(int, str(value).split(":"))
    return hour * 60 + minute


def check_cost_model() -> None:
    base = load_cost_estimates(ROOT / "data/test/city_costs.csv")
    expanded = expand_test_cost_estimates(base)
    keys = [(x.origin, x.city, x.days) for x in expanded]
    assert len(keys) == len(set(keys)), "В сметах обнаружены дубли"
    originals = {(x.origin, x.city, x.days): x for x in base}
    derived = {(x.origin, x.city, x.days): x for x in expanded}
    assert len(base) == 225, f"Ожидалось 225 базовых смет, получено {len(base)}"
    assert len(expanded) == 675, f"Ожидалось 675 смет, получено {len(expanded)}"
    for original in base:
        origin, city = original.origin, original.city
        assert derived[(origin, city, 2)] == original
        one = derived[(origin, city, 1)]
        three = derived[(origin, city, 3)]
        assert one.costs.round_trip_tickets == original.costs.round_trip_tickets
        assert three.costs.round_trip_tickets == original.costs.round_trip_tickets
        assert one.costs.nights == 0
        assert three.costs.nights == (0 if origin == city else 2)
        assert abs(one.costs.activities_and_food - original.costs.activities_and_food / 2) < 0.01
        assert abs(three.costs.activities_and_food - original.costs.activities_and_food * 1.5) < 0.01
        assert "УЧЕБНЫЙ пересчет" in one.source and "УЧЕБНЫЙ пересчет" in three.source
    print(f"OK: {len(base)} двухдневных + {len(expanded)-len(base)} расчетных = {len(expanded)} смет")


def check_route(city: str, days: int, *, family: bool = False) -> None:
    # Высокий бюджет не мешает проверке оптимизатора: лимит расходов на
    # посещения остается отдельно 10 000 ₽ за ВСЮ поездку.
    profile = TravelProfile(
        origin="Москва", days=days, total_budget=100_000,
        tickets_budget=80_000, activity_budget=10_000,
        interests=("Культура и история",), with_children=family,
        weather="обычная", risk_attitude="нейтральный",
    )
    result = analyze_trip(
        profile, selected_city=city, build_route=True, feedback_extra_min=0,
    )
    if result.route is None:
        raise AssertionError("Маршрут не построен: " + " | ".join(result.notices))
    route = result.route
    plan = route.plan
    assert set(route.schedules) == set(range(1, days + 1)), "Нет расписания на каждый день"
    assert set(plan["day"]) == set(range(1, days + 1)), "Неполное распределение по дням"
    # Достопримечательности не повторяются; кафе можно посетить несколько раз
    # в разные дни, но не дважды в один день.
    nonfood = plan.loc[plan["category"] != "food", "object_id"].tolist()
    assert len(nonfood) == len(set(nonfood)), "Достопримечательность повторилась"
    for day in range(1, days + 1):
        selected = plan.loc[plan["day"] == day]
        assert not selected["object_id"].duplicated().any(), f"Повтор в день {day}"
        food = selected.loc[selected["category"] == "food"]
        assert len(food) >= route.meals_per_day, f"Недостаточно приемов пищи в день {day}"
        schedule = route.schedules[day]
        assert not schedule.empty, f"Пустое расписание в день {day}"
        assert minutes(schedule.iloc[-1]["Прибытие"]) <= 20 * 60
        starts = []
        for name in food["name"]:
            match = schedule.loc[schedule["Точка"] == name, "Начало"]
            assert len(match) == 1, f"Кафе {name} отсутствует в расписании дня {day}"
            starts.append(minutes(match.iloc[0]))
        assert sum(720 <= v <= 900 for v in starts) == 1, "Не найден один обед 12:00–15:00"
        if route.meals_per_day == 2:
            assert sum(1020 <= v <= 1140 for v in starts) == 1, "Не найден ужин 17:00–19:00"
    assert route.activities_cost <= route.activity_limit + 0.01, "Лимит расходов превышен"
    assert route.final_budget["fits_limits"], "Не соблюден общий бюджет"
    print(f"OK: {city:18s} | {days} дн. | {'семья' if family else 'обычный'} | "
          f"объектов {len(plan):2d} | приемов пищи/день {route.meals_per_day} | "
          f"смета {route.final_budget['total']:.0f} ₽")


def main() -> None:
    check_cost_model()
    # Двухдневный вариант уже проверен ранее; здесь ключевые новые случаи.
    failures = []
    for days in (1, 3):
        for city in CITIES:
            try:
                check_route(city, days)
            except Exception as exc:
                failures.append((city, days, False, str(exc)))
                print(f"ОШИБКА: {city}, {days} дн.: {exc}")
    # Отдельная проверка наиболее трудного случая: 3 дня, шесть приемов пищи,
    # при наличии всего четырех кафе в Казани.
    try:
        check_route("Казань", 3, family=True)
    except Exception as exc:
        failures.append(("Казань", 3, True, str(exc)))
        print(f"ОШИБКА: Казань, 3 дня, семья: {exc}")
    if failures:
        raise SystemExit(f"Не пройдено проверок маршрутов: {len(failures)}")
    print("\nOK: все сценарии 1–3 дней и семейное питание прошли проверки")


if __name__ == "__main__":
    main()
