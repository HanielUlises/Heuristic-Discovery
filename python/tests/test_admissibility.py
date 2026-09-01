"""Verification of heuristics against exact goal distances.

The aggregation tests run on synthetic verifier output; the tests that need
the compiled ``hd_verify`` are skipped when it has not been built.
"""

import pytest

from hd.admissibility import (
    AdmissibilityReport,
    InstanceVerification,
    StateSpaceSummary,
    VerificationRun,
    heuristic_label,
    summary_table,
)
from hd.candidate import BASELINE_HEURISTICS, HeuristicCandidate
from hd.domains import blocksworld


# --- synthetic fixtures ---------------------------------------------------

def make_space(states: int = 100, complete: bool = True, optimal: float = 8.0):
    return StateSpaceSummary(
        status="complete" if complete else "state_limit",
        complete=complete,
        states=states,
        transitions=states * 3,
        goal_states=1,
        dead_end_states=0,
        max_h_star=12.0,
        initial_h_star=optimal,
        runtime_seconds=0.01,
    )


def make_report(violations: int = 0, informedness: float = 0.5, consistent: bool = True):
    return AdmissibilityReport(
        admissible=violations == 0,
        consistent=consistent and violations == 0,
        goal_aware=True,
        non_negative=True,
        states_checked=100,
        transitions_checked=300,
        dead_end_states=0,
        goal_states=1,
        admissibility_violations=violations,
        consistency_violations=0 if consistent and violations == 0 else 7,
        max_excess=3.5 if violations else 0.0,
        max_ratio=2.0 if violations else 0.0,
        max_consistency_excess=0.0,
        max_goal_value=0.0,
        min_value=0.0,
        max_value=9.0,
        mean_informedness=informedness,
        min_informedness=0.1,
        mean_error=2.0,
    )


def make_run(*instances: InstanceVerification, heuristics=("h",)) -> VerificationRun:
    return VerificationRun(
        timestamp="2024-01-01T00:00:00Z",
        limits={"max_states": 1000},
        build={"version": "test"},
        heuristics=list(heuristics),
        instances=list(instances),
    )


# --- aggregation ----------------------------------------------------------

def test_a_clean_run_summarises_as_admissible():
    run = make_run(
        InstanceVerification("i0", "i0.task", make_space(), {"h": make_report()}),
        InstanceVerification("i1", "i1.task", make_space(), {"h": make_report(informedness=0.7)}),
    )
    summary = run.summary("h")
    assert summary.admissible
    assert summary.consistent
    assert summary.instances_checked == 2
    assert summary.instances_skipped == 0
    assert summary.total_violations == 0
    assert summary.mean_informedness == pytest.approx(0.6)
    assert summary.witness is None
    assert summary.verdict() == "admissible, consistent"


def test_one_violating_instance_condemns_the_heuristic():
    run = make_run(
        InstanceVerification("i0", "i0.task", make_space(), {"h": make_report()}),
        InstanceVerification("i1", "i1.task", make_space(), {"h": make_report(violations=4)}),
    )
    summary = run.summary("h")
    assert not summary.admissible
    assert summary.violating_instances == 1
    assert summary.total_violations == 4
    assert summary.max_excess == pytest.approx(3.5)
    assert summary.verdict() == "inadmissible"


def test_a_truncated_instance_is_skipped_rather_than_passed():
    run = make_run(
        InstanceVerification("small", "small.task", make_space(), {"h": make_report()}),
        InstanceVerification("large", "large.task", make_space(complete=False), {}),
    )
    summary = run.summary("h")
    assert summary.instances_checked == 1
    assert summary.instances_skipped == 1
    assert [i.instance for i in run.skipped()] == ["large"]
    assert "large" in summary_table(run)


def test_a_run_with_nothing_enumerable_reports_no_verdict():
    run = make_run(InstanceVerification("large", "large.task", make_space(complete=False), {}))
    summary = run.summary("h")
    assert summary.instances_checked == 0
    assert not summary.admissible
    assert summary.verdict() == "unchecked"


def test_an_unverified_heuristic_is_an_error():
    run = make_run(InstanceVerification("i0", "i0.task", make_space(), {"h": make_report()}))
    with pytest.raises(KeyError):
        run.summary("other")


def test_labels_distinguish_baselines_from_candidates():
    assert heuristic_label("goal_count") == "goal_count"
    assert heuristic_label(HeuristicCandidate(name="best", weights={"relaxed_layers": 1.0})) == "best"


# --- integration ----------------------------------------------------------

@pytest.fixture(scope="module")
def instance(tmp_path_factory):
    path = tmp_path_factory.mktemp("instances") / "blocks-4.task"
    path.write_text(blocksworld.generate(num_blocks=4, seed=17).to_task_text("blocks-4"))
    return path


def test_the_baselines_survive_verification(verifier, instance):
    run = verifier.verify([instance], list(BASELINE_HEURISTICS))
    assert run.instances[0].checked
    for name in ("zero", "goal_count", "relaxed_layers"):
        report = run.instances[0].reports[name]
        assert report.admissible, report.admissibility_witnesses
        assert report.consistent
        assert report.goal_aware
        assert report.non_negative
    # Better-informed baselines recover more of h*, and none of them exceed it.
    informedness = [run.summary(n).mean_informedness for n in ("zero", "goal_count")]
    assert informedness[0] == 0.0
    assert 0.0 < informedness[1] <= 1.0


def test_the_landmark_bound_is_admissible_and_better_informed(verifier, instance):
    run = verifier.verify([instance], ["relaxed_layers", "landmark_cost"])
    landmarks = run.instances[0].reports["landmark_cost"]
    layers = run.instances[0].reports["relaxed_layers"]

    assert landmarks.admissible, landmarks.admissibility_witnesses
    assert landmarks.goal_aware
    assert landmarks.non_negative
    assert landmarks.mean_informedness > layers.mean_informedness
    # Pricing landmarks from the state alone does not give consistency: the
    # landmark set can change non-monotonically along a transition.
    assert not landmarks.consistent


def test_an_inflated_heuristic_is_falsified_with_a_witness(verifier, instance):
    inflated = HeuristicCandidate(name="inflated", weights={"unsatisfied_goals": 50.0})
    run = verifier.verify([instance], ["relaxed_layers", inflated])
    summary = run.summary("inflated")

    assert not summary.admissible
    assert summary.total_violations > 0
    assert summary.witness is not None
    assert summary.witness.h > summary.witness.h_star
    assert summary.witness.state  # the propositions true at the violating state
    assert run.summary("relaxed_layers").admissible


def test_the_phase_one_weights_are_not_goal_aware(verifier, instance):
    # Weights on features that are non-zero at a goal state cannot vanish
    # there, so the candidate overestimates the goal itself.
    candidate = HeuristicCandidate(
        name="phase1", weights={"achieved_goals": 1.0, "true_propositions": 1.0}
    )
    report = verifier.verify([instance], [candidate]).instances[0].reports["phase1"]
    assert not report.goal_aware
    assert report.max_goal_value > 0.0
    assert not report.admissible


def test_the_oracle_agrees_with_optimal_search(verifier, planner, instance):
    # Two independent computations of the same quantity: backward Dijkstra over
    # the enumerated space, and A* with an admissible heuristic.
    space = verifier.verify([instance], ["relaxed_layers"]).instances[0].state_space
    run = planner.run([instance], heuristic="relaxed_layers", search="astar")
    assert space.initial_h_star == pytest.approx(run.runs[0].solution_cost)


def test_the_enumeration_ceiling_is_honoured(verifier, instance):
    run = verifier.verify([instance], ["goal_count"], max_states=10)
    assert not run.instances[0].checked
    assert run.instances[0].state_space.status == "state_limit"
    assert run.instances[0].reports == {}


def test_verifying_nothing_is_an_error(verifier, instance):
    with pytest.raises(ValueError):
        verifier.verify([], ["goal_count"])
    with pytest.raises(ValueError):
        verifier.verify([instance], [])
    with pytest.raises(ValueError):
        verifier.verify([instance], ["goal_count", "goal_count"])
