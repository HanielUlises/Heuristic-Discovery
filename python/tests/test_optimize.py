"""Optimisers: determinism, bounds, and improvement tracking."""

import pytest

from hd.candidate import HeuristicCandidate
from hd.objective import ObjectiveValue
from hd.optimize import (
    SearchSpace,
    default_search_space,
    grid_search,
    is_degenerate,
    local_search,
    random_search,
)

SPACE = SearchSpace({"unsatisfied_goals": (0.0, 4.0), "applicable_actions": (0.0, 4.0)})


def quadratic(candidate: HeuristicCandidate) -> ObjectiveValue:
    """A deterministic surrogate objective with its optimum at (2, 1)."""
    a = candidate.weights.get("unsatisfied_goals", 0.0)
    b = candidate.weights.get("applicable_actions", 0.0)
    value = (a - 2.0) ** 2 + (b - 1.0) ** 2
    return ObjectiveValue(value=value, components={"expanded": value}, coverage=1.0)


def test_search_space_rejects_unknown_features_and_empty_intervals():
    with pytest.raises(ValueError, match="unknown feature"):
        SearchSpace({"not_a_feature": (0.0, 1.0)})
    with pytest.raises(ValueError, match="empty interval"):
        SearchSpace({"relaxed_sum": (1.0, 0.0)})


def test_samples_and_clipping_respect_the_bounds():
    import random

    rng = random.Random(0)
    for _ in range(50):
        for feature, value in SPACE.sample(rng).items():
            low, high = SPACE.bounds[feature]
            assert low <= value <= high
    clipped = SPACE.clip({"unsatisfied_goals": 99.0, "applicable_actions": -99.0})
    assert clipped == {"unsatisfied_goals": 4.0, "applicable_actions": 0.0}


def test_random_search_is_reproducible_under_a_seed():
    first = random_search(SPACE, quadratic, iterations=8, seed=42)
    second = random_search(SPACE, quadratic, iterations=8, seed=42)
    assert first.best_candidate.weights == second.best_candidate.weights
    assert [t.objective.value for t in first.history] == [t.objective.value for t in second.history]


def test_different_seeds_explore_differently():
    first = random_search(SPACE, quadratic, iterations=8, seed=1)
    second = random_search(SPACE, quadratic, iterations=8, seed=2)
    assert first.best_candidate.weights != second.best_candidate.weights


def test_the_reported_best_is_the_minimum_of_the_trajectory():
    result = random_search(SPACE, quadratic, iterations=16, seed=3)
    assert result.num_evaluations == 16
    assert result.best_objective.value == min(t.objective.value for t in result.history)
    assert sum(1 for t in result.history if t.accepted) >= 1


def test_an_initial_candidate_is_evaluated_first():
    initial = HeuristicCandidate({"unsatisfied_goals": 2.0, "applicable_actions": 1.0})
    result = random_search(SPACE, quadratic, iterations=4, seed=0, initial=initial)
    assert result.history[0].candidate.weights == initial.weights
    assert result.best_objective.value == pytest.approx(0.0)  # the optimum was supplied


def test_local_search_improves_on_its_starting_point():
    start = HeuristicCandidate({"unsatisfied_goals": 0.0, "applicable_actions": 4.0})
    result = local_search(SPACE, quadratic, iterations=60, seed=5, initial=start)
    assert result.best_objective.value < quadratic(start).value
    for feature, (low, high) in SPACE.bounds.items():
        assert low <= result.best_candidate.weights[feature] <= high


def test_local_search_is_reproducible_under_a_seed():
    first = local_search(SPACE, quadratic, iterations=10, seed=7)
    second = local_search(SPACE, quadratic, iterations=10, seed=7)
    assert first.best_candidate.weights == second.best_candidate.weights


def test_grid_search_covers_the_product_of_its_axes():
    result = grid_search(SPACE, quadratic, points_per_axis=3)
    assert result.num_evaluations == 9
    corners = {tuple(sorted(t.candidate.weights.items())) for t in result.history}
    assert len(corners) == 9
    assert result.best_objective.value == min(t.objective.value for t in result.history)


def test_grid_search_with_one_point_uses_the_box_centre():
    result = grid_search(SPACE, quadratic, points_per_axis=1)
    assert result.num_evaluations == 1
    assert result.best_candidate.weights == {"unsatisfied_goals": 2.0, "applicable_actions": 2.0}


def test_trials_are_reported_to_the_callback():
    seen = []
    random_search(SPACE, quadratic, iterations=5, seed=0, on_trial=seen.append)
    assert len(seen) == 5
    assert [t.iteration for t in seen] == [0, 1, 2, 3, 4]


def test_results_serialise_to_plain_data():
    result = random_search(SPACE, quadratic, iterations=3, seed=0)
    data = result.to_dict()
    assert data["optimizer"] == "random_search"
    assert data["num_evaluations"] == 3
    assert len(data["history"]) == 3
    assert set(data["best_candidate"]["weights"]) == set(SPACE.features)


def test_default_space_spans_every_feature_non_negatively():
    space = default_search_space()
    assert all(low == 0.0 and high > 0.0 for low, high in space.bounds.values())


def test_degeneracy_detection():
    assert is_degenerate(HeuristicCandidate({"unsatisfied_goals": 0.0}))
    assert not is_degenerate(HeuristicCandidate({"unsatisfied_goals": 0.5}))
