import pandas as pd
from src.knowledge.facts import TEST_PROFILES, TOURIST, build_knowledge_base
from src.knowledge.inference import explain, forward_chain, objects_with
from src.knowledge.rules import RULES
from src.knowledge.schema import (
    FIGURES_DIR,
    class_attributes,
    draw_schema,
    extract_schema,
)

DEFAULT_MEALS_PER_DAY = 1   # умолчание, если правила не вывели другое


def tourist_value(kb, relation, default=None):
    """Значение атрибута туриста, выведенное правилами (или умолчание)"""
    found = kb.query([(TOURIST, relation, "?v")])
    return found[0]["?v"] if found else default


def run_profile(profile: dict):
    """Строит базу знаний для профиля и выполняет прямой вывод"""
    kb = build_knowledge_base(profile)
    explanations, log = forward_chain(kb, RULES)
    return kb, explanations, log


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    rules_by_name = {rule.name: rule for rule in RULES}

    rows = {}
    details = {}
    results = {}
    for name, profile in TEST_PROFILES.items():
        kb, explanations, log = run_profile(profile)
        results[name] = (kb, explanations)
        fired = sorted({r for counter in log for r in counter},
                       key=lambda r: int(r[1:]))
        details[name] = {
            "Рекомендованы": objects_with(kb, "рекомендован"),
            "Обязательно к посещению": objects_with(kb, "обязательно_к_посещению"),
            "Только при хорошей погоде": objects_with(
                kb, "рекомендован_при_хорошей_погоде"),
            "Приоритет для семьи": objects_with(kb, "приоритет_для_семьи"),
        }
        rows[name] = {
            "Выведено фактов": len(explanations),
            "Итераций": len(log),
            "Сработали правила": ", ".join(fired),
            **{key: len(value) for key, value in details[name].items()},
            "Приемов пищи в день": tourist_value(
                kb, "приемов_пищи_в_день", DEFAULT_MEALS_PER_DAY),
            "Критерий выбора направления": tourist_value(
                kb, "критерий_выбора_направления"),
        }

    print("Сравнение результатов вывода для трех профилей:")
    print(pd.DataFrame(rows).to_string())
    print()

    for name, groups in details.items():
        print(f"=== {name} ===")
        for title, names in groups.items():
            if names:
                print(f"  {title} ({len(names)}): " + "; ".join(names))
        print()

    kb, explanations = results["Семья с детьми"]
    print("Объяснение: почему Кремлевская набережная для семьи - "
          "только при хорошей погоде")
    for line in explain(("embankment", "рекомендован_при_хорошей_погоде",
                         TOURIST), explanations, rules_by_name):
        print(line)
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    schema = extract_schema(kb)
    draw_schema(schema, class_attributes(kb),
                FIGURES_DIR / "kb_schema_inferred.png",
                title="Схема базы знаний после логического вывода",
                derived={fact[1] for fact in explanations})
    print("Схема с выведенными отношениями сохранена в "
          f"{FIGURES_DIR / 'kb_schema_inferred.png'}")
