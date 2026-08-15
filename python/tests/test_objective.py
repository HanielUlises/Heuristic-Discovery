"""The objective: normalisation, weighting and the treatment of failure."""

import pytest
from conftest import make_metrics, make_run

from hd.objective import Objective, ObjectiveValue, ObjectiveWeights


def reference():
    return make_run(make_metrics("a", expanded=100), make_metrics("b", expanded=200))


def test_the_reference_scores_exactly_one():
    objective = Objective.from_reference_run(reference())
    assert objective(reference()).value == pytest.approx(1.0, abs=1e-6)


def test_halving_expansions_halves_the_objective():
    objective = Objective.from_reference_run(reference())
    better = make_run(make_metrics("a", expanded=50), make_metrics("b", expanded=100))
    assert objective(better).value == pytest.approx(0.5, abs=1e-6)


def test_normalisation_is_per_instance_not_aggregate():
    # One instance is ten times cheaper, the other ten times more expensive.
    # A per-instance normalisation scores this as (0.1 + 10) / 2, whereas
    # normalising totals would hide the regression.
    objective = Objective.from_reference_run(reference())
    mixed = make_run(make_metrics("a", expanded=10), make_metrics("b", expanded=2000))
    assert objective(mixed).components["expanded"] == pytest.approx(5.05, abs=1e-3)


def test_weights_select_which_terms_contribute():
    runs = make_run(make_metrics("a", expanded=50, runtime=0.2, cost=20.0))
    ref = make_run(make_metrics("a", expanded=100, runtime=0.1, cost=10.0))

    only_expanded = Objective.from_reference_run(ref, ObjectiveWeights(expanded=1.0))
    assert only_expanded(runs).value == pytest.approx(0.5, abs=1e-6)

    only_runtime = Objective.from_reference_run(
        ref, ObjectiveWeights(expanded=0.0, runtime=1.0)
    )
    assert only_runtime(runs).value == pytest.approx(2.0, abs=1e-4)

    only_cost = Objective.from_reference_run(ref, ObjectiveWeights(expanded=0.0, cost=1.0))
    assert only_cost(runs).value == pytest.approx(2.0, abs=1e-6)


def test_unsolved_instances_are_penalised_and_excluded_from_cost():
    ref = reference()
    objective = Objective.from_reference_run(ref, ObjectiveWeights(expanded=0.0, unsolved=10.0))
    partial = make_run(make_metrics("a", expanded=100), make_metrics("b", solved=False))
    value = objective(partial)
    assert value.coverage == pytest.approx(0.5)
    assert value.value == pytest.approx(5.0)
    assert value.components["cost"] == pytest.approx(1.0)  # solved instance only


def test_instances_missing_from_the_reference_are_scored_unnormalised():
    objective = Objective.from_reference_run(reference())
    unknown = make_run(make_metrics("z", expanded=7))
    assert objective(unknown).components["expanded"] == pytest.approx(7.0)


def test_zero_reference_does_not_divide_by_zero():
    ref = make_run(make_metrics("a", expanded=0, runtime=0.0, cost=0.0))
    objective = Objective.from_reference_run(ref)
    value = objective(make_run(make_metrics("a", expanded=0, runtime=0.0, cost=0.0)))
    assert value.value == pytest.approx(1.0)


def test_empty_runs_are_rejected():
    with pytest.raises(ValueError):
        Objective().evaluate([])


def test_objective_value_is_serialisable_and_float_like():
    value = ObjectiveValue(0.5, {"expanded": 0.5}, coverage=1.0)
    assert float(value) == 0.5
    assert value.to_dict()["components"]["expanded"] == 0.5


def test_weights_round_trip():
    weights = ObjectiveWeights(expanded=1.0, runtime=0.5, cost=0.25, unsolved=8.0)
    assert ObjectiveWeights.from_dict(weights.to_dict()) == weights
