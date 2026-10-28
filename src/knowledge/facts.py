import pandas as pd
from src.knowledge.kb import KnowledgeBase
from src.multicriteria.ahp import (
    TEST_COMPARISON_MATRIX,
    calculate_weights,
    evaluate_alternatives,
)
from src.multicriteria.alternatives import PROJECT_ROOT, load_alternatives
from src.optimization.objects import load_objects
from src.optimization.travel import cluster_objects
from src.uncertainty.criteria import find_dominance, risk_std
from src.uncertainty.scenarios import (
    build_payoff_matrix,
    build_scenarios,
    load_uncertainty,
)

TOURIST = "турист"
CITY_OF_OBJECTS = "Казань"

# Таксономия: категория объекта → тематика (экспертное знание)
TAXONOMY = {
    "history": "Культура и история",
    "religion": "Культура и история",
    "architecture": "Культура и история",
    "museum": "Культура и история",
    "park": "Прогулки и природа",
    "walk": "Прогулки и природа",
    "food": "Гастрономия",
    "entertainment": "Развлечения",
    "excursion": "Загородные поездки",
}

# Тестовые профили туристов (ЛПР)
TEST_PROFILES = {
    "Студент": {
        "interests": ["Культура и история"],
        "budget": 1000,
        "days": 2,
        "with_children": False,
        "first_visit": True,
        "weather": "обычная",
        "risk_attitude": "осторожный",
    },
    "Семья с детьми": {
        "interests": ["Развлечения", "Прогулки и природа"],
        "budget": 5000,
        "days": 2,
        "with_children": True,
        "first_visit": True,
        "weather": "хуже",
        "risk_attitude": "нейтральный",
    },
    "Гурман": {
        "interests": ["Гастрономия", "Культура и история"],
        "budget": 3000,
        "days": 2,
        "with_children": False,
        "first_visit": False,
        "weather": "лучше",
        "risk_attitude": "оптимист",
    },
}


def add_taxonomy(kb: KnowledgeBase) -> None:
    for category, theme in TAXONOMY.items():
        kb.add(category, "тип", "Категория")
        kb.add(theme, "тип", "Тематика")
        kb.add(category, "входит_в", theme)


def add_objects(kb: KnowledgeBase, objects: pd.DataFrame) -> None:
    """Факты об объектах из kazan_objects.csv"""
    # Кластеры названы по своему главному объекту (kremlin), поэтому
    # добавляем префикс: у каждой сущности должен быть уникальный идентификатор
    clusters = "кластер_" + cluster_objects(objects)
    for obj, row in objects.iterrows():
        kb.add(obj, "тип", "Объект")
        kb.add(obj, "название", row["name"])
        kb.add(obj, "находится_в", CITY_OF_OBJECTS)
        kb.add(obj, "относится_к", row["category"])
        kb.add(obj, "в_кластере", clusters[obj])
        kb.add(obj, "в_помещении", bool(row["indoor"]))
        kb.add(obj, "цена", int(row["price"]))
        kb.add(obj, "длительность", int(row["visit_min"]))
        kb.add(obj, "привлекательность", round(float(row["attractiveness"]), 2))
        kb.add(clusters[obj], "тип", "Кластер")


def add_cities(kb: KnowledgeBase) -> None:
    """Факты о направлениях: оценка AHP, доминирование и риск"""
    weights = calculate_weights(TEST_COMPARISON_MATRIX)
    cities, scores = load_alternatives(PROJECT_ROOT / "data" / "test" / "destinations.csv")
    uncertainty = load_uncertainty(PROJECT_ROOT / "data" / "test" / "uncertainty.csv", cities)
    scenarios = build_scenarios()
    payoff = build_payoff_matrix(cities, scores, uncertainty, scenarios, weights)
    risk = risk_std(payoff, scenarios["probability"].to_numpy())
    for city, score in zip(cities, evaluate_alternatives(scores, weights)):
        kb.add(city, "тип", "Город")
        kb.add(city, "оценка_AHP", round(float(score), 3))
        kb.add(city, "риск_СКО", round(float(risk[city]), 3))
    for city, dominators in find_dominance(payoff).items():
        for winner in dominators:
            kb.add(winner, "доминирует", city)


def add_profile(kb: KnowledgeBase, profile: dict) -> None:
    """Факты о туристе (ЛПР)"""
    kb.add(TOURIST, "тип", "Турист")
    for theme in profile["interests"]:
        kb.add(TOURIST, "интересуется", theme)
    kb.add(TOURIST, "бюджет", profile["budget"])
    kb.add(TOURIST, "дней", profile["days"])
    kb.add(TOURIST, "с_детьми", profile["with_children"])
    kb.add(TOURIST, "впервые_в_городе", profile["first_visit"])
    kb.add(TOURIST, "ожидает_погоду", profile["weather"])
    kb.add(TOURIST, "отношение_к_риску", profile["risk_attitude"])


def build_knowledge_base(profile: dict) -> KnowledgeBase:
    """Собирает базу фактов: справочники, данные этапов 2-4 и профиль ЛПР"""
    kb = KnowledgeBase()
    add_taxonomy(kb)
    objects = load_objects(PROJECT_ROOT / "data" / "raw" / "kazan_objects.csv").set_index("id")
    add_objects(kb, objects)
    add_cities(kb)
    add_profile(kb, profile)
    return kb


if __name__ == "__main__":
    kb = build_knowledge_base(TEST_PROFILES["Студент"])
    print(f"Фактов в базе: {len(kb)}")
    print("Фактов по отношениям:")
    for relation, count in kb.relations().items():
        print(f"  {relation}: {count}")
    print()

    print("Пример: все факты об объекте kulsharifmosque")
    for b in sorted(kb.query([("kulsharifmosque", "?r", "?v")]),
                    key=lambda b: b["?r"]):
        print(f"  (kulsharifmosque, {b['?r']}, {b['?v']})")
    print()

    print("Запрос: объекты из тематики, которой интересуется турист")
    print("  (турист, интересуется, ?t) ∧ (?c, входит_в, ?t) ∧ (?o, относится_к, ?c)")
    found = kb.query([
        (TOURIST, "интересуется", "?t"),
        ("?c", "входит_в", "?t"),
        ("?o", "относится_к", "?c"),
    ])
    print(f"  найдено объектов: {len(found)}")
    print()

    dominated = kb.query([("Казань", "доминирует", "?city")])
    print(f"Казань доминирует направлений: {len(dominated)} из 14")
