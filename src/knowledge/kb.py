from collections import defaultdict


def is_variable(term) -> bool:
    """Переменная в шаблоне - строка, начинающаяся с '?', например '?obj'"""
    return isinstance(term, str) and term.startswith("?")


class KnowledgeBase:
    """
    База фактов в виде семантической сети: множество триплетов
    (субъект, отношение, объект). Поддерживает поиск по шаблонам
    с переменными
    """
    def __init__(self):
        self.facts = set()
        self._by_relation = defaultdict(set)

    def add(self, subject, relation, obj) -> bool:
        """Добавляет факт. Возвращает True, если факт новый"""
        fact = (subject, relation, obj)
        if fact in self.facts:
            return False
        self.facts.add(fact)
        self._by_relation[relation].add(fact)
        return True

    def __len__(self):
        return len(self.facts)

    def relations(self) -> dict:
        """Число фактов по каждому отношению"""
        return {r: len(f) for r, f in sorted(self._by_relation.items())}

    def _match(self, pattern, fact, bindings):
        """Сопоставляет один шаблон с одним фактом с учетом уже связанных переменных"""
        result = dict(bindings)
        for term, value in zip(pattern, fact):
            if is_variable(term):
                if term in result and result[term] != value:
                    return None
                result[term] = value
            elif term != value:
                return None
        return result

    def query(self, patterns, bindings=None) -> list:
        """
        Все наборы значений переменных, при которых выполняются
        ВСЕ шаблоны одновременно (соединение по общим переменным).
        Пример: [("?o", "относится_к", "?c"), ("?c", "входит_в", "Гастрономия")]
        """
        bindings = bindings or {}
        if not patterns:
            return [bindings]
        first, rest = patterns[0], patterns[1:]
        candidates = (self._by_relation.get(first[1], set())
                      if not is_variable(first[1]) else self.facts)
        results = []
        for fact in candidates:
            new = self._match(first, fact, bindings)
            if new is not None:
                results.extend(self.query(rest, new))
        return results
