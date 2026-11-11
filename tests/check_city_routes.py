"""Этап 8: проверка двухдневных маршрутов для трех новых городов.

Запуск из корня travel-dds: python tests/check_city_routes.py
Использует существующий сервис, оптимизатор PuLP/CBC и учебные сметы.
Не меняет исходные данные и другие файлы проекта.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dss.service import CityRouteConfig, analyze_trip
from src.optimization.objects import load_objects
from src.recommendations.profile import TravelProfile


CONFIGS = {
    "Нижний Новгород": CityRouteConfig(
        catalog_path=ROOT / "data" / "raw" / "nizhny_novgorod_objects.csv",
        city_center=(56.3269, 44.0065),
        start_coordinates=(56.3269, 44.0065),
    ),
    "Калининград": CityRouteConfig(
        catalog_path=ROOT / "data" / "raw" / "kaliningrad_objects.csv",
        city_center=(54.706129, 20.511063),
        start_coordinates=(54.706129, 20.511063),
    ),
    "Сочи": CityRouteConfig(
        catalog_path=ROOT / "data" / "raw" / "sochi_objects.csv",
        city_center=(43.5855, 39.7231),
        start_coordinates=(43.5855, 39.7231),
        max_distance_km=60.0,
    ),
}


def check_route(city: str, route, days: int) -> list[str]:
    """Проверяет ключевые инварианты результата, не подменяя MILP-тесты."""
    problems = []
    plan = route.plan
    if plan.empty:
        problems.append("план пуст")
        return problems

    if len(plan) != len(set(plan["object_id"])):
        problems.append("объект присутствует в маршруте более одного раза")

    for day in range(1, days + 1):
        day_plan = plan.loc[plan["day"] == day]
        if day_plan.empty:
            problems.append(f"день {day} остался пустым")
        else:
            food_count = int((day_plan["category"] == "food").sum())
            if food_count < route.meals_per_day:
                problems.append(
                    f"день {day}: мест питания {food_count}, "
                    f"требуется {route.meals_per_day}"
                )
        schedule = route.schedules.get(day)
        if schedule is None or schedule.empty:
            problems.append(f"нет расписания для дня {day}")
        else:
            last_arrival = str(schedule.iloc[-1]["Прибытие"])
            try:
                return_time = datetime.strptime(last_arrival, "%H:%M").time()
                if return_time > datetime.strptime("20:00", "%H:%M").time():
                    problems.append(
                        f"день {day}: возвращение в {last_arrival}, позже 20:00"
                    )
            except ValueError:
                problems.append(f"день {day}: некорректное время {last_arrival}")

    actual_cost = float(plan["price"].sum())
    if abs(actual_cost - route.activities_cost) > 0.01:
        problems.append("сумма цен объектов не равна расходам в результате")
    if route.activities_cost > route.activity_limit + 0.01:
        problems.append("превышен лимит расходов на объекты")
    if not route.final_budget["fits_limits"]:
        problems.append("нарушены ограничения общей сметы")
    return problems


def main() -> int:
    profile = TravelProfile(
        origin="Москва",
        days=2,
        total_budget=100_000,
        tickets_budget=80_000,
        activity_budget=3_000,
        interests=("Культура и история",),
        weather="обычная",
        risk_attitude="нейтральный",
    )

    failures = []
    print("Проверяем маршруты на 2 дня из Москвы, бюджет объектов — 3 000 ₽.")
    print("Стоимость билетов и проживания в data/test/ — учебная.\n")

    for city, config in CONFIGS.items():
        print("=" * 64)
        print("ГОРОД:", city)
        try:
            if not Path(config.catalog_path).is_file():
                raise FileNotFoundError(f"Не найден {config.catalog_path}")

            objects = load_objects(
                config.catalog_path,
                city_center=config.city_center,
                max_distance_km=config.max_distance_km,
            )
            print(
                f"Каталог: {len(objects)} объектов; "
                f"питание: {int((objects['category'] == 'food').sum())}"
            )

            # Передаем настройки явно: проверка Сочи работает, даже если он
            # еще не добавлен в постоянный CITY_ROUTE_CONFIGS сервиса.
            result = analyze_trip(
                profile,
                selected_city=city,
                build_route=True,
                feedback_extra_min=0,  # одна итерация — быстрая первичная проверка
                city_configs={city: config},
            )
            if result.city_budget is not None:
                print(f"Предварительная смета: {result.city_budget['total']:.0f} ₽")

            if result.route is None:
                failures.append(city)
                print("НЕ УДАЛОСЬ ПОСТРОИТЬ МАРШРУТ")
                for note in result.notices:
                    print("-", note)
                continue

            route = result.route
            print(f"Объектов в маршруте: {len(route.plan)}")
            print(f"Персональная оценка: {route.personal_score:.2f}")
            print(f"Расходы на объекты: {route.activities_cost:.0f} ₽")
            print(f"Итоговая смета: {route.final_budget['total']:.0f} ₽")
            print(f"Приемов пищи в день: {route.meals_per_day}")
            for day, schedule in sorted(route.schedules.items()):
                print(f"\nДЕНЬ {day}")
                print(schedule.to_string(index=False))

            problems = check_route(city, route, profile.days)
            if problems:
                failures.append(city)
                print("\nНАЙДЕНЫ ПРОБЛЕМЫ:")
                for issue in problems:
                    print("-", issue)
            else:
                print("\nOK: маршрут соответствует проверенным ограничениям")

        except Exception as exc:
            failures.append(city)
            print(f"ОШИБКА: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 64)
    if failures:
        print("Не прошли проверку:", ", ".join(failures))
        return 1
    print("OK: маршруты всех трех городов прошли проверку")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
