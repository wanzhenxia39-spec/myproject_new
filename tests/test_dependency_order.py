from itertools import permutations

import numpy as np
import pandas as pd

from zshar.semantics.dependency_order import (
    directional_dependency_matrix,
    optimal_dependency_order,
    order_objective,
    semantic_dependency_order,
)


def test_directional_dependency_detects_one_way_prediction():
    # A identifies all four rows, so it fully predicts B. B groups rows and
    # cannot fully predict A, giving a strictly directional score.
    prototypes = pd.DataFrame(
        {
            "A": ["a0", "a1", "a2", "a3"],
            "B": ["low", "low", "high", "high"],
        }
    )
    matrix, entropies = directional_dependency_matrix(prototypes)
    assert entropies["A"] == 2.0
    assert entropies["B"] == 1.0
    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 0.5
    assert np.diag(matrix).tolist() == [0.0, 0.0]


def test_constant_target_has_zero_incoming_dependency():
    prototypes = pd.DataFrame(
        {"varying": ["L", "M", "H"], "constant": ["L", "L", "L"]}
    )
    matrix, _ = directional_dependency_matrix(prototypes)
    assert matrix[0, 1] == 0.0


def test_subset_dp_matches_exhaustive_optimum_and_tie_breaking():
    matrix = np.asarray(
        [
            [0.0, 0.9, 0.2, 0.5],
            [0.1, 0.0, 0.8, 0.4],
            [0.7, 0.3, 0.0, 0.6],
            [0.2, 0.3, 0.1, 0.0],
        ]
    )
    order, score = optimal_dependency_order(matrix)
    candidates = [
        (order_objective(candidate, matrix), candidate)
        for candidate in permutations(range(4))
    ]
    optimum = max(value for value, _ in candidates)
    expected = min(candidate for value, candidate in candidates if np.isclose(value, optimum))
    assert np.isclose(score, optimum)
    assert tuple(order) == expected


def test_semantic_order_returns_exact_reverse():
    result = semantic_dependency_order(
        pd.DataFrame(
            {
                "A": ["L", "M", "H", "L"],
                "B": ["L", "L", "H", "L"],
                "C": ["H", "M", "L", "H"],
            },
            index=["seen1", "seen2", "seen3", "seen4"],
        )
    )
    assert result["reverse"] == result["forward"][::-1]
    assert set(result["forward"]) == {"A", "B", "C"}
