from collections import Counter
from src.knowledge.facts import TEST_PROFILES, TOURIST, build_knowledge_base
from src.knowledge.kb import KnowledgeBase, is_variable
from src.knowledge.rules import RULES, Rule, matches


def substitute(pattern: tuple, bindings: dict) -> tuple:
    """Подставляет значения переменных в шаблон триплета"""
    return tuple(bindings[t] if is_variable(t) else t for t in pattern)


def forward_chain(kb: KnowledgeBase, rules: list, max_iterations: int = 50):
    """
    Прямой вывод «волнами»: на каждой итерации все правила применяются
    к фактам, известным в начале итерации; новые факты добавляются
    в конце итерации. Вывод останавливается, когда новых фактов нет
    Возвращает:
      explanations - {выведенный факт: (правило, факты-посылки, итерация)}
      log - список Counter: сколько новых фактов дало каждое правило
            на каждой итерации
    """
    explanations = {}
    log = []
    for iteration in range(1, max_iterations + 1):
        pending = {}
        for rule in rules:
            for bindings in matches(kb, rule):
                premises = [substitute(c, bindings) for c in rule.conditions]
                for conclusion in rule.conclusions:
                    fact = substitute(conclusion, bindings)
                    if fact not in kb.facts and fact not in pending:
                        pending[fact] = (rule.name, premises, iteration)
        if not pending:
            break
        for fact, reason in pending.items():
            kb.add(*fact)
            explanations[fact] = reason
        log.append(Counter(rule for rule, _, _ in pending.values()))
    return explanations, log


def explain(fact: tuple, explanations: dict, rules_by_name: dict,
            depth: int = 0) -> list:
    """
    Дерево объяснения факта: каким правилом он выведен и из каких
    посылок; посылки раскрываются рекурсивно до исходных фактов
    """
    indent = "    " * depth
    text = f"({fact[0]}, {fact[1]}, {fact[2]})"
    if fact not in explanations:
        return [f"{indent}• {text} — исходный факт"]
    rule_name, premises, iteration = explanations[fact]
    lines = [f"{indent}• {text} — выведен правилом {rule_name} "
             f"(итерация {iteration}): {rules_by_name[rule_name].description}"]
    for premise in premises:
        lines.extend(explain(premise, explanations, rules_by_name, depth + 1))
    return lines


def objects_with(kb: KnowledgeBase, relation: str) -> list:
    """Названия объектов, для которых выведено отношение relation к туристу"""
    found = kb.query([("?o", relation, TOURIST), ("?o", "название", "?n")])
    return sorted(b["?n"] for b in found)


if __name__ == "__main__":
    rules_by_name = {rule.name: rule for rule in RULES}
    kb = build_knowledge_base(TEST_PROFILES["Студент"])
    initial = len(kb)
    explanations, log = forward_chain(kb, RULES)

    print(f"Исходных фактов: {initial}, выведено новых: {len(explanations)}, "
          f"итераций с новыми фактами: {len(log)}")
    for number, counter in enumerate(log, start=1):
        details = ", ".join(f"{r}: {n}" for r, n in sorted(
            counter.items(), key=lambda item: int(item[0][1:])))
        print(f"  Итерация {number}: {details}")
    print()

    recommended = objects_with(kb, "рекомендован")
    must_see = objects_with(kb, "обязательно_к_посещению")
    print(f"Рекомендовано объектов: {len(recommended)}")
    print("  " + "; ".join(recommended))
    print(f"Обязательно к посещению: {len(must_see)}")
    print("  " + "; ".join(must_see))
    print()

    criterion = kb.query([(TOURIST, "критерий_выбора_направления", "?c")])
    remaining = [b["?g"] for b in kb.query([("?g", "тип", "Город")])
                 if (b["?g"], "исключено_из_рассмотрения", True) not in kb.facts]
    print(f"Критерий выбора направления: {criterion[0]['?c']}")
    print(f"Направлений после исключения доминируемых: {remaining}")
    print()

    print("Объяснение вывода: почему рекомендована мечеть Кул-Шариф")
    for line in explain(("kulsharifmosque", "рекомендован", TOURIST),
                        explanations, rules_by_name):
        print(line)
