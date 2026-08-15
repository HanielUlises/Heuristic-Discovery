"""Candidate heuristics: representation, rendering and serialisation."""

import json

import pytest

from hd.candidate import FEATURE_NAMES, HeuristicCandidate, baseline_candidate, zero_candidate


def test_weights_are_coerced_to_float():
    candidate = HeuristicCandidate({"unsatisfied_goals": 2})
    assert candidate.weights == {"unsatisfied_goals": 2.0}
    assert isinstance(candidate.weights["unsatisfied_goals"], float)


def test_unknown_features_are_rejected():
    with pytest.raises(ValueError, match="unknown feature"):
        HeuristicCandidate({"not_a_feature": 1.0})


def test_spec_round_trips_through_the_planner_syntax():
    candidate = HeuristicCandidate({"unsatisfied_goals": 1.82, "applicable_actions": 0.37})
    spec = candidate.to_spec()
    assert spec.startswith("linear:")
    terms = dict(t.split("=") for t in spec[len("linear:"):].split(","))
    assert float(terms["unsatisfied_goals"]) == pytest.approx(1.82)
    assert float(terms["applicable_actions"]) == pytest.approx(0.37)


def test_negligible_weights_are_dropped_from_the_spec():
    candidate = HeuristicCandidate({"unsatisfied_goals": 1.0, "achieved_goals": 0.0})
    assert candidate.to_spec() == "linear:unsatisfied_goals=1.0"


def test_an_all_zero_candidate_degenerates_to_the_zero_heuristic():
    assert zero_candidate().to_spec() == "zero"
    assert zero_candidate().to_equation() == "h(s) = 0"


def test_equation_is_readable_and_signed():
    candidate = HeuristicCandidate({"unsatisfied_goals": 1.82, "applicable_actions": -0.37})
    equation = candidate.to_equation()
    assert "h(s) = 1.82 * unsatisfied_goals(s)" in equation
    assert "- 0.37 * applicable_actions(s)" in equation


def test_json_round_trip(tmp_path):
    candidate = HeuristicCandidate(
        {"unsatisfied_goals": 1.5, "relaxed_sum": 0.25}, name="discovered", metadata={"seed": 7}
    )
    path = candidate.save(tmp_path / "candidate.json")
    restored = HeuristicCandidate.load(path)
    assert restored.weights == candidate.weights
    assert restored.name == "discovered"
    assert restored.metadata["seed"] == 7
    assert json.loads(path.read_text())["weights"]["relaxed_sum"] == 0.25


def test_yaml_round_trip(tmp_path):
    candidate = HeuristicCandidate({"unsatisfied_goals": 1.5}, name="discovered")
    path = candidate.save(tmp_path / "candidate.yaml")
    restored = HeuristicCandidate.load(path)
    assert restored.weights["unsatisfied_goals"] == pytest.approx(1.5)
    assert restored.name == "discovered"


def test_normalisation_preserves_the_greedy_ordering():
    candidate = HeuristicCandidate({"unsatisfied_goals": 2.0, "relaxed_sum": 1.0})
    normalised = candidate.normalised()
    assert max(abs(v) for v in normalised.weights.values()) == pytest.approx(1.0)
    ratio = normalised.weights["unsatisfied_goals"] / normalised.weights["relaxed_sum"]
    assert ratio == pytest.approx(2.0)


def test_zero_candidate_normalisation_is_a_no_op():
    assert zero_candidate().normalised().weights == zero_candidate().weights


def test_baseline_names_are_validated():
    assert baseline_candidate("goal_count") == "goal_count"
    with pytest.raises(ValueError):
        baseline_candidate("nonexistent")


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
