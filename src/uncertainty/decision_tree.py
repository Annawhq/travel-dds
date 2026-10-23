import networkx as nx

# Тестовые параметры задачи покупки авиабилета (руб)
# Значения не основаны на реальных данных
TEST_TICKET_PARAMS = {
    "city": "Казань",
    "price_now": 6000,             # невозвратный тариф сейчас
    "refundable_price": 7200,      # возвратный тариф сейчас
    "refund_fee": 0,               # удержание при возврате возвратного тарифа
    "p_plans_change": 0.15,        # вероятность, что поездка не состоится
    "price_change": [              # (состояние, множитель цены, вероятность)
        ("цены ниже", 0.9, 0.2),
        ("цены базовые", 1.0, 0.5),
        ("цены выше", 1.4, 0.3),
    ],
    "train_price": 4500,           # альтернатива: поезд
    "train_time_penalty": 3000,    # денежный эквивалент потерянного дня поездки
}


def build_ticket_tree(params: dict) -> nx.DiGraph:
    """
    Строит дерево решений "когда и какой билет покупать"
    Узлы: decision (решение ЛПР), chance (случайное событие)
    terminal (исход с расходами ЛПР в рублях)
    """
    tree = nx.DiGraph()

    def add(node, kind, label, cost=None):
        tree.add_node(node, kind=kind, label=label, cost=cost)

    def link(parent, child, label, probability=None):
        tree.add_edge(parent, child, label=label, probability=probability)

    p_change = params["p_plans_change"]
    p_go = 1 - p_change

    add("D0", "decision", "Как купить билет?")

    # Стратегия 1: невозвратный билет сейчас
    add("C_nonref", "chance", "Поездка состоится?")
    link("D0", "C_nonref", "Невозвратный сейчас")
    add("T_nonref_go", "terminal", "Поездка", params["price_now"])
    add("T_nonref_cancel", "terminal", "Отмена, деньги потеряны",
        params["price_now"])
    link("C_nonref", "T_nonref_go", "да", p_go)
    link("C_nonref", "T_nonref_cancel", "нет", p_change)

    # Стратегия 2: возвратный билет сейчас
    add("C_ref", "chance", "Поездка состоится?")
    link("D0", "C_ref", "Возвратный сейчас")
    add("T_ref_go", "terminal", "Поездка", params["refundable_price"])
    add("T_ref_cancel", "terminal", "Отмена, возврат", params["refund_fee"])
    link("C_ref", "T_ref_go", "да", p_go)
    link("C_ref", "T_ref_cancel", "нет", p_change)

    # Стратегия 3: подождать и купить позже
    add("C_wait", "chance", "Поездка состоится?")
    link("D0", "C_wait", "Подождать")
    add("T_wait_cancel", "terminal", "Отмена, билет не куплен", 0)
    link("C_wait", "T_wait_cancel", "нет", p_change)

    add("C_price", "chance", "Как изменится цена?")
    link("C_wait", "C_price", "да", p_go)

    train_cost = params["train_price"] + params["train_time_penalty"]
    for i, (state, multiplier, probability) in enumerate(params["price_change"]):
        decision = f"D_late_{i}"
        add(decision, "decision", "Чем ехать?")
        link("C_price", decision, state, probability)

        plane = f"T_plane_{i}"
        train = f"T_train_{i}"
        add(plane, "terminal", "Самолет", params["price_now"] * multiplier)
        add(train, "terminal", "Поезд (с учетом потери времени)", train_cost)
        link(decision, plane, "самолет")
        link(decision, train, "поезд")

    if not nx.is_arborescence(tree):
        raise ValueError("Построенный граф не является деревом.")
    for node in tree.nodes:
        if tree.nodes[node]["kind"] == "chance":
            total = sum(tree.edges[node, c]["probability"]
                        for c in tree.successors(node))
            if abs(total - 1) > 1e-9:
                raise ValueError(f"В узле {node} сумма вероятностей = {total}")
    return tree


def roll_back(tree: nx.DiGraph, node: str = "D0") -> float:
    """
    Свертка дерева справа налево. Для chance-узла считается ожидаемое
    значение, для decision-узла выбирается ветвь с минимальными
    ожидаемыми расходами. Результаты записываются в атрибуты узлов
    """
    data = tree.nodes[node]

    if data["kind"] == "terminal":
        data["value"] = data["cost"]
    elif data["kind"] == "chance":
        data["value"] = sum(
            tree.edges[node, child]["probability"] * roll_back(tree, child)
            for child in tree.successors(node)
        )
    else:
        options = {child: roll_back(tree, child)
                   for child in tree.successors(node)}
        best = min(options, key=options.get)
        data["value"] = options[best]
        data["best"] = best

    return data["value"]


def print_tree(tree: nx.DiGraph, node: str = "D0", depth: int = 0,
               edge_label: str = "") -> None:
    """Текстовый вывод дерева с ожидаемыми расходами в каждом узле"""
    data = tree.nodes[node]
    marks = {"decision": "[Р]", "chance": "(С)", "terminal": " → "}
    prefix = "    " * depth + (f"{edge_label}: " if edge_label else "")
    print(f"{prefix}{marks[data['kind']]} {data['label']} "
          f"= {data['value']:.0f} руб.")

    for child in tree.successors(node):
        edge = tree.edges[node, child]
        label = edge["label"]
        if edge["probability"] is not None:
            label += f" (p={edge['probability']:.2f})"
        if data.get("best") == child:
            label += " ✔"
        print_tree(tree, child, depth + 1, label)


if __name__ == "__main__":
    params = TEST_TICKET_PARAMS
    tree = build_ticket_tree(params)
    expected_cost = roll_back(tree)

    print(f"Дерево решений: покупка билета, направление {params['city']}")
    print(f"Узлов: {tree.number_of_nodes()}, ребер: {tree.number_of_edges()}")
    print()
    print_tree(tree)
    print()

    print("Ожидаемые расходы по стратегиям:")
    for child in tree.successors("D0"):
        print(f"  {tree.edges['D0', child]['label']:<20} "
              f"{tree.nodes[child]['value']:.0f} руб.")
    best = tree.nodes["D0"]["best"]
    print(f"Оптимальная стратегия: {tree.edges['D0', best]['label']} "
          f"({expected_cost:.0f} руб.)")
