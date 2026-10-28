import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from src.knowledge.facts import TEST_PROFILES, build_knowledge_base
from src.knowledge.kb import KnowledgeBase
from src.multicriteria.alternatives import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def extract_schema(kb: KnowledgeBase) -> set:
    """
    Схема онтологии, извлеченная из самих фактов:
    если (a, r, b) - факт, и у a и b известен тип,
    то в схеме есть связь (тип(a), r, тип(b))
    """
    types = {s: o for (s, r, o) in kb.facts if r == "тип"}
    return {
        (types[s], r, types[o])
        for (s, r, o) in kb.facts
        if r != "тип" and s in types and o in types
    }


def class_attributes(kb: KnowledgeBase) -> dict:
    """Атрибуты каждого класса: отношения, ведущие к значениям без типа"""
    types = {s: o for (s, r, o) in kb.facts if r == "тип"}
    attributes = {}
    for (s, r, o) in kb.facts:
        if r != "тип" and s in types and o not in types:
            attributes.setdefault(types[s], set()).add(r)
    return attributes


def draw_schema(schema: set, attributes: dict, path,
                title: str = "Схема базы знаний (извлечена из фактов)",
                derived: set | None = None) -> None:
    """
    Рисует схему: классы - прямоугольники с атрибутами, связи - стрелки
    derived - отношения, выведенные правилами: такие связи рисуются
    пунктиром, а атрибуты помечаются звездочкой
    """
    derived = derived or set()
    graph = nx.DiGraph()
    loops = {}
    for source, relation, target in sorted(schema):
        if source == target:
            loops.setdefault(source, []).append(relation)
        elif graph.has_edge(source, target):
            graph[source][target]["relations"].append(relation)
        else:
            graph.add_edge(source, target, relations=[relation])

    graph.add_nodes_from(loops)
    for node in attributes:
        if node in graph or attributes[node]:
            graph.add_node(node)

    positions = {
        "Турист": (0, 2), "Тематика": (2, 2), "Категория": (4, 2),
        "Город": (2, 0), "Объект": (4, 0), "Кластер": (6, 0),
    }

    def mark(relation):
        return relation + ("*" if relation in derived else "")

    labels = {}
    for node in graph.nodes:
        text = node.upper()
        if attributes.get(node):
            text += "\n" + "\n".join(mark(a) for a in sorted(attributes[node]))
        if loops.get(node):
            text += "\n↻ " + ", ".join(mark(r) for r in loops[node])
        labels[node] = text

    base_edges, derived_edges = [], []
    for u, v, data in graph.edges(data=True):
        if all(r in derived for r in data["relations"]):
            derived_edges.append((u, v))
        else:
            base_edges.append((u, v))

    plt.figure(figsize=(13, 7.5))
    nx.draw_networkx_edges(graph, positions, edgelist=base_edges, arrows=True,
                           arrowsize=20, node_size=12000, edge_color="#4a6fa5")
    nx.draw_networkx_edges(graph, positions, edgelist=derived_edges,
                           arrows=True, arrowsize=20, node_size=12000,
                           edge_color="#c0392b", style="dashed")
    edge_labels = {(u, v): "\n".join(d["relations"])
                   for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        graph, positions, edge_labels, font_size=8, rotate=False,
        font_color="#1f3b63", bbox=dict(fc="white", ec="none", alpha=0.9),
    )
    nx.draw_networkx_labels(
        graph, positions, labels, font_size=8,
        bbox=dict(boxstyle="round,pad=0.6", fc="#eef4fb", ec="#4a6fa5"),
    )
    if derived:
        plt.figtext(0.5, 0.03, "Пунктир и * — отношения и атрибуты, "
                    "выведенные продукционными правилами",
                    ha="center", fontsize=9, color="#c0392b")
    plt.title(title)
    plt.margins(x=0.12, y=0.25)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    kb = build_knowledge_base(TEST_PROFILES["Студент"])
    schema = extract_schema(kb)
    attributes = class_attributes(kb)

    print("Связи между классами:")
    for source, relation, target in sorted(schema):
        print(f"  {source} —{relation}→ {target}")
    print()
    print("Атрибуты классов:")
    for cls, attrs in sorted(attributes.items()):
        print(f"  {cls}: {', '.join(sorted(attrs))}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    draw_schema(schema, attributes, FIGURES_DIR / "kb_schema.png")
    print()
    print(f"Схема сохранена в {FIGURES_DIR / 'kb_schema.png'}")
