"""Этапы 7–8: персонализированные рекомендации объектов разных городов.
По умолчанию доступен существующий каталог Казани. Для другого города
необходимо передать путь к его каталогу и координаты центра города.
Результат - рейтинг объектов
"""

from pathlib import Path

import pandas as pd

from src.knowledge.facts import TAXONOMY, TOURIST, add_profile, add_taxonomy
from src.knowledge.inference import forward_chain
from src.knowledge.kb import KnowledgeBase
from src.knowledge.rules import RULES
from src.multicriteria.alternatives import PROJECT_ROOT
from src.optimization.objects import load_objects
from src.recommendations.profile import TravelProfile


KAZAN_CATALOG = PROJECT_ROOT / "data" / "raw" / "kazan_objects.csv"
RESULT_COLUMNS = [
    "city", "object_id", "name", "category", "theme", "score",
    "interest_score", "attractiveness", "price", "visit_min",
    "indoor", "kb_flags", "explanation",
]


def _knowledge_flags(objects: pd.DataFrame, profile: TravelProfile) -> dict[str, set[str]]:
    """Получает выводы базы знаний этапа 5 для текущего профиля
    Формируем только факты, нужные правилам об объектах. Факты о
    доминировании городов (R12) здесь намеренно не используем
    """
    kb = KnowledgeBase()
    add_taxonomy(kb)
    add_profile(kb, profile.to_knowledge_profile())

    for item in objects.itertuples(index=False):
        kb.add(item.id, "тип", "Объект")
        kb.add(item.id, "относится_к", item.category)
        kb.add(item.id, "цена", int(item.price))
        kb.add(item.id, "в_помещении", bool(item.indoor))
        kb.add(item.id, "привлекательность", round(float(item.attractiveness), 2))

    forward_chain(kb, RULES)

    relations = (
        "рекомендован", "обязательно_к_посещению", "приоритет_для_семьи",
        "рекомендован_при_хорошей_погоде",
    )
    result = {}
    for item in objects.itertuples(index=False):
        flags = {
            relation for relation in relations
            if (item.id, relation, TOURIST) in kb.facts
        }
        if (item.id, "примечание", "подходит семье, но оценки посетителей невысокие") in kb.facts:
            flags.add("примечание_о_низкой_оценке")
        result[item.id] = flags
    return result


def recommend_places(
    city: str,
    profile: TravelProfile,
    top_n: int | None = 10,
    *,
    catalog_path: str | Path | None = None,
    city_center: tuple[float, float] | None = None,
    max_distance_km: float = 60.0,
) -> pd.DataFrame:
    """Ранжирует объекты города с учетом профиля ЛПР.
    Старый вызов recommend_places("Казань", profile) работает без изменений.
    Для других городов обязательны catalog_path (CSV того же формата,
    что kazan_objects.csv) и city_center=(широта, долгота).
    Фильтры исключают отдельные слишком дорогие объекты и уличные
    объекты при плохой погоде. Бюджет всего маршрута проверяет оптимизатор.
    """
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1
    ):
        raise ValueError("top_n должен быть положительным целым числом или None")

    if not isinstance(city, str) or not city.strip():
        raise ValueError("Укажи непустое название города")
    city = city.strip()

    if catalog_path is None:
        if city != "Казань":
            raise ValueError(
                f"Для города '{city}' нужно передать catalog_path "
                "и city_center. Каталог по умолчанию есть только у Казани."
            )
        catalog = KAZAN_CATALOG
    else:
        catalog = Path(catalog_path)

    if city != "Казань" and city_center is None:
        raise ValueError(
            f"Для города '{city}' укажи city_center=(широта, долгота), "
            "чтобы проверить координаты объектов."
        )

    objects = load_objects(
        catalog,
        city_center=city_center,
        max_distance_km=max_distance_km,
    )
    flags = _knowledge_flags(objects, profile)
    rows = []

    for item in objects.itertuples(index=False):
        # Лимит — на все развлечения и питание; здесь проверяем только
        # невозможность оплатить один объект, а не весь маршрут.
        if profile.activity_budget is not None and item.price > profile.activity_budget:
            continue
        if profile.weather == "хуже" and not bool(item.indoor):
            continue

        theme = TAXONOMY[item.category]
        if not profile.interests:
            match = 3.0  # нет указанных интересов — нейтральная оценка
        else:
            match = 5.0 if theme in profile.interests else 1.0

        if profile.activity_budget is None:
            affordability = 3.0  # лимит не задан — цена нейтральна
        elif profile.activity_budget == 0:
            affordability = 5.0  # после фильтра остались только бесплатные
        else:
            affordability = 5.0 - 4.0 * item.price / profile.activity_budget

        expert = flags[item.id]
        # Малый вес экспертной оценки обеспечивает влияние правил БЗ,
        # но не позволяет им полностью переопределить интересы туриста.
        if "обязательно_к_посещению" in expert:
            expert_score = 5.0
        elif "приоритет_для_семьи" in expert:
            expert_score = 4.5
        elif "рекомендован" in expert:
            expert_score = 4.0
        else:
            expert_score = 2.5
        # Все компоненты лежат в [1, 5], веса суммируются к 1
        score = (
            0.45 * match + 0.30 * float(item.attractiveness)
            + 0.15 * affordability + 0.10 * expert_score
        )

        reasons = [
            f"Тематика: {theme} (соответствие интересам {match:.0f}/5)",
            f"привлекательность {item.attractiveness:.2f}/5",
            f"цена {item.price:.0f} руб.",
        ]
        if "обязательно_к_посещению" in expert:
            reasons.append("база знаний: знаковая достопримечательность при первом посещении")
        if "приоритет_для_семьи" in expert:
            reasons.append("база знаний: приоритет для семьи с детьми")
        if "рекомендован" in expert:
            reasons.append("соответствует правилу рекомендации R6")
        if "примечание_о_низкой_оценке" in expert:
            reasons.append("база знаний: приоритет для семьи, но оценка невысока")

        rows.append({
            "city": city,
            "object_id": item.id,
            "name": item.name,
            "category": item.category,
            "theme": theme,
            "score": float(score),
            "interest_score": float(match),
            "attractiveness": float(item.attractiveness),
            "price": float(item.price),
            "visit_min": int(item.visit_min),
            "indoor": bool(item.indoor),
            "kb_flags": "; ".join(sorted(expert)),
            "explanation": "; ".join(reasons) + ".",
        })

    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if result.empty:
        return result
    result = result.sort_values(
        ["score", "object_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    return result if top_n is None else result.head(top_n).copy()


if __name__ == "__main__":
    profiles = {
        "Культура": TravelProfile(
            interests=("Культура и история",), activity_budget=1000
        ),
        "Гастрономия": TravelProfile(
            interests=("Гастрономия",), activity_budget=3000,
            first_visit=False,
        ),
        "Семья в дождь": TravelProfile(
            interests=("Развлечения", "Прогулки и природа"),
            activity_budget=5000, with_children=True, weather="хуже",
        ),
    }
    for title, profile in profiles.items():
        all_places = recommend_places("Казань", profile, top_n=None)
        print(f"\n{title}: допустимых объектов {len(all_places)} из 40")
        print(all_places[["name", "score", "price"]].head(5)
              .round(2).to_string(index=False))
