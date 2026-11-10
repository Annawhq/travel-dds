"""Этап 7. Загрузка смет из CSV для проверки бюджета поездки
В CSV можно хранить тестовые сметы и наблюдаемые цены, обязательно указывая происхождение данных
"""
import csv
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable, Mapping
from src.recommendations.budget import TripCosts
from src.recommendations.city_budget import (
    CityCostEstimate,
    recommend_cities_with_budget,
)
from src.recommendations.profile import TravelProfile


REQUIRED_COLUMNS = (
    "origin", "city", "days", "round_trip_tickets",
    "price_per_night", "nights", "activities_and_food",
    "local_transport", "reserve_fraction", "data_type",
    "source", "checked_on",
)


def parse_cost_estimates(rows: Iterable[Mapping[str, str]],) -> list[CityCostEstimate]:
    """Преобразует строки CSV в сметы для city_budget.py
    data_type:
      test - учебная смета;
      observed - цена, записанная из внешнего источника
    Для observed обязательна дата проверки
    Пустые цены не заменяются нулем
    """
    result = []
    seen = set()
    for line_number, row in enumerate(rows, start=2):
        try:
            missing = [
                col for col in REQUIRED_COLUMNS
                if col not in row
            ]
            if missing:
                raise ValueError(f"нет столбцов: {missing}")
            data_type = row["data_type"].strip().lower()
            if data_type not in ("test", "observed"):
                raise ValueError(
                    "data_type должен быть test или observed"
                )
            source = row["source"].strip()
            if not source:
                raise ValueError("обязателен источник source")
            checked_raw = row["checked_on"].strip()
            checked = (
                date.fromisoformat(checked_raw)
                if checked_raw else None
            )
            if data_type == "observed" and checked is None:
                raise ValueError(
                    "для observed обязательна checked_on"
                )
            if checked is not None and checked > date.today():
                raise ValueError(
                    "дата проверки не может быть в будущем"
                )
            estimate = CityCostEstimate(
                origin=row["origin"].strip(),
                city=row["city"].strip(),
                days=int(row["days"]),
                costs=TripCosts(
                    round_trip_tickets=float(
                        row["round_trip_tickets"]
                    ),
                    price_per_night=float(
                        row["price_per_night"]
                    ),
                    nights=int(row["nights"]),
                    activities_and_food=float(
                        row["activities_and_food"]
                    ),
                    local_transport=float(
                        row["local_transport"]
                    ),
                    reserve_fraction=float(
                        row["reserve_fraction"]
                    ),
                ),
                source=(
                    f"[{'ТЕСТ' if data_type == 'test' else 'НАБЛЮДЕНИЕ'}] "
                    f"{source}"
                ),
                checked_on=checked,
            )
            key = (
                estimate.origin,
                estimate.city,
                estimate.days,
            )
            if key in seen:
                raise ValueError(
                    "повторная смета для той же пары "
                    "городов и числа дней"
                )
            seen.add(key)
            result.append(estimate)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Ошибка CSV в строке {line_number}: {exc}"
            ) from exc
    return result


def load_cost_estimates(path: str | Path,) -> list[CityCostEstimate]:
    """Читает локальный CSV с ценами в рублях"""
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "CSV пуст: нет заголовка"
            )
        missing = [
            col for col in REQUIRED_COLUMNS
            if col not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"В CSV отсутствуют столбцы: {missing}"
            )
        return parse_cost_estimates(reader)


if __name__ == "__main__":
    # Демонстрационные сметы предыдущего шага
    demo_csv = """origin,city,days,round_trip_tickets,price_per_night,nights,activities_and_food,local_transport,reserve_fraction,data_type,source,checked_on
Москва,Казань,2,8000,3500,1,3000,800,0.10,test,учебный пример,
Москва,Калининград,2,15000,4000,1,3000,1000,0.10,test,учебный пример,
"""
    estimates = parse_cost_estimates(
        csv.DictReader(StringIO(demo_csv))
    )
    profile = TravelProfile(
        origin="Москва",
        interests=("Культура и история",),
        days=2,
        total_budget=18_000,
        tickets_budget=9_000,
        activity_budget=3_000,
    )
    result = recommend_cities_with_budget(
        profile,
        estimates,
        top_n=None,
        only_affordable=False,
    )
    print(f"Смет загружено: {len(estimates)}")
    print(
        "Направлений без сметы:",
        (
            result["budget_status"] == "смета отсутствует"
        ).sum(),
    )
    print(
        result.set_index("city").loc[
            ["Казань", "Калининград", "Сочи"],
            ["estimated_total", "budget_status"],
        ].to_string()
    )
