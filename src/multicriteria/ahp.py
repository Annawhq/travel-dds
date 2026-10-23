import numpy as np
from src.multicriteria.alternatives import PROJECT_ROOT, load_alternatives

RI_VALUES = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def calculate_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Рассчитывает веса критериев методом главного собственного вектора.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    max_index = np.argmax(eigenvalues.real)

    principal_eigenvector = eigenvectors[:, max_index].real

    weights = principal_eigenvector / principal_eigenvector.sum()

    return weights


def calculate_consistency_ratio(
    matrix: np.ndarray,
    weights: np.ndarray) -> float:
    """
    Рассчитывает коэффициент согласованности CR.
    """
    n = matrix.shape[0]

    weighted_sum = matrix @ weights

    consistency_vector = weighted_sum / weights

    lambda_max = consistency_vector.mean()

    consistency_index = (lambda_max - n) / (n - 1)

    random_index = RI_VALUES.get(n)

    if random_index is None:
        raise ValueError(
            f"Для матрицы размером {n}x{n} значение RI не задано."
        )

    if random_index == 0:
        return 0.0

    consistency_ratio = consistency_index / random_index

    return consistency_ratio


def evaluate_alternatives(
    alternatives_matrix: np.ndarray,
    weights: np.ndarray) -> np.ndarray:
    """
    Рассчитывает итоговые оценки альтернатив
    на основе оценок по критериям и весов критериев.
    """
    return alternatives_matrix @ weights


if __name__ == "__main__":
    criteria = [
        "Стоимость",
        "Погода",
        "Транспортная доступность",
        "Туристическая привлекательность",
        "Соответствие интересам",
    ]

    comparison_matrix = np.array([
        [1,   3,   3,   2,   2],
        [1/3, 1,   1, 1/3, 1/3],
        [1/3, 1,   1, 1/3, 1/3],
        [1/2, 3,   3,   1,   1],
        [1/2, 3,   3,   1,   1],
    ], dtype=float)

    weights = calculate_weights(comparison_matrix)
    consistency_ratio = calculate_consistency_ratio(
        comparison_matrix,
        weights
    )
    print("Веса критериев:")
    for criterion, weight in zip(criteria, weights):
        print(f"{criterion}: {weight:.4f}")
    print()

    alternatives, alternatives_matrix = load_alternatives(
        PROJECT_ROOT / "data" / "test" / "destinations.csv"
    )

    print(f"CR = {consistency_ratio:.4f}")
    if consistency_ratio < 0.1:
        print("Матрица согласована.")
    else:
        print("Матрица недостаточно согласована.")

    scores = evaluate_alternatives(
        alternatives_matrix,
        weights
    )
    ranking = sorted(
        zip(alternatives, scores),
        key=lambda item: item[1],
        reverse=True
    )
    print()
    print("Рейтинг направлений:")
    for position, (city, score) in enumerate(ranking, start=1):
        print(f"{position}. {city}: {score:.4f}")
