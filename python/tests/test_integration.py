"""Integration: Python -> C++ planner -> JSON -> Python evaluation.

These tests require the compiled planner; they are skipped when it is absent.
"""

import pytest

from hd.benchmark import BenchmarkSuite, summarise
from hd.candidate import FEATURE_NAMES, HeuristicCandidate
from hd.domains import blocksworld
from hd.experiment import Experiment, build_record
from hd.objective import Objective, ObjectiveWeights
from hd.optimize import SearchSpace, random_search
from hd.planner import PlannerError


@pytest.fixture(scope="module")
def instance(tmp_path_factory):
    """A single small Blocksworld task written to disk."""
    path = tmp_path_factory.mktemp("instances") / "blocks-4.task"
    path.write_text(blocksworld.generate(num_blocks=4, seed=17).to_task_text("blocks-4"))
    return path


def test_the_python_feature_list_matches_the_engine(planner):
    engine = [f["name"] for f in planner.features()]
    assert engine == list(FEATURE_NAMES)


def test_build_information_is_reported(planner):
    build = planner.build_info()
    assert build["version"]
    assert build["compiler"]
    assert int(build["cxx_standard"]) >= 202002  # C++20 or later


def test_a_baseline_run_produces_well_formed_metrics(planner, instance):
    run = planner.run([instance], heuristic="goal_count", search="gbfs")
    assert run.search == "gbfs"
    assert len(run.runs) == 1
    metrics = run.runs[0]
    assert metrics.instance == "blocks-4"
    assert metrics.solved
    assert metrics.solution_cost == metrics.solution_length  # unit costs
    assert metrics.expanded >= 1
    assert metrics.generated >= metrics.expanded
    assert metrics.runtime_seconds >= 0.0


def test_a_candidate_heuristic_reaches_the_engine_unchanged(planner, instance):
    candidate = HeuristicCandidate({"unsatisfied_goals": 1.82, "applicable_actions": 0.37})
    run = planner.run([instance], heuristic=candidate)
    assert run.heuristic["kind"] == "linear"
    assert run.heuristic["weights"]["unsatisfied_goals"] == pytest.approx(1.82)
    assert run.heuristic["weights"]["applicable_actions"] == pytest.approx(0.37)


def test_astar_returns_an_optimal_plan(planner, instance):
    optimal = planner.run([instance], heuristic="zero", search="astar")
    informed = planner.run([instance], heuristic="relaxed_layers", search="astar")
    assert optimal.runs[0].solution_cost == informed.runs[0].solution_cost
    assert informed.runs[0].expanded <= optimal.runs[0].expanded


def test_emitted_plans_are_executable_action_names(planner, instance):
    run = planner.run([instance], heuristic="goal_count", emit_plan=True)
    plan = run.runs[0].plan
    assert len(plan) == run.runs[0].solution_length
    assert all(a.split("_")[0] in {"pickup", "putdown", "stack", "unstack"} for a in plan)


def test_repeated_invocations_are_deterministic(planner, instance):
    candidate = HeuristicCandidate({"unsatisfied_goals": 1.0, "relaxed_sum": 0.5})
    first = planner.run([instance], heuristic=candidate).runs[0]
    second = planner.run([instance], heuristic=candidate).runs[0]
    assert (first.expanded, first.generated, first.solution_cost) == (
        second.expanded,
        second.generated,
        second.solution_cost,
    )


def test_the_expansion_budget_is_enforced(planner, instance):
    run = planner.run([instance], heuristic="zero", max_expansions=5)
    metrics = run.runs[0]
    assert not metrics.solved
    assert metrics.status == "expansion_limit"
    assert metrics.expanded <= 5
    assert metrics.solution_cost is None


def test_planner_errors_are_surfaced(planner, tmp_path):
    with pytest.raises(PlannerError):
        planner.run([tmp_path / "does-not-exist.task"])


def test_several_instances_are_evaluated_in_one_invocation(planner, tmp_path):
    paths = []
    for blocks, seed in ((3, 1), (4, 2), (5, 3)):
        path = tmp_path / f"blocks-{blocks}.task"
        path.write_text(blocksworld.generate(blocks, seed).to_task_text(f"blocks-{blocks}"))
        paths.append(path)
    run = planner.run(paths, heuristic="goal_count")
    assert [m.instance for m in run.runs] == ["blocks-3", "blocks-4", "blocks-5"]
    assert summarise(run.runs).num_solved == 3


def test_end_to_end_discovery_loop(planner, benchmark_manifest, tmp_path):
    """Python proposes candidates, the engine plans, Python scores and improves."""
    suite = BenchmarkSuite.load(benchmark_manifest).subset(6)
    experiment = Experiment(
        suite=suite,
        planner=planner,
        weights=ObjectiveWeights(expanded=1.0, unsolved=10.0),
        max_expansions=20_000,
        time_limit=10.0,
        seed=0,
    )

    # The reference configuration scores exactly one by construction.
    assert experiment.evaluate("goal_count").objective.value == pytest.approx(1.0, abs=1e-3)

    space = SearchSpace({"unsatisfied_goals": (0.0, 4.0), "relaxed_sum": (0.0, 4.0)})
    result = random_search(space, experiment.scorer(), iterations=6, seed=1)

    assert result.num_evaluations == 6
    assert result.best_objective.value == min(t.objective.value for t in result.history)
    assert set(result.best_candidate.weights) == set(space.features)

    evaluation = experiment.evaluate(result.best_candidate)
    assert evaluation.summary.num_instances == len(suite)
    assert evaluation.objective.value == pytest.approx(result.best_objective.value)

    record = build_record(
        "integration", experiment, {"discovered": evaluation}, result, space
    ).save(tmp_path)
    data = Experiment and __import__("json").loads(record.read_text())
    assert data["schema"] == "hd.experiment/1"
    assert data["config"]["num_instances"] == len(suite)
    assert data["environment"]["planner_build"]["version"]
    assert data["optimization"]["seed"] == 1
    assert len(data["optimization"]["history"]) == 6
    assert data["instances"] and all(i.startswith("instances/") for i in data["instances"])


def test_a_saved_candidate_reproduces_its_metrics(planner, instance, tmp_path):
    candidate = HeuristicCandidate({"unsatisfied_goals": 1.3, "relaxed_sum": 0.7})
    first = planner.run([instance], heuristic=candidate)
    restored = HeuristicCandidate.load(candidate.save(tmp_path / "c.json"))
    second = planner.run([instance], heuristic=restored)

    objective = Objective.from_reference_run(first)
    assert objective(second).value == pytest.approx(1.0, abs=1e-6)
    assert second.runs[0].expanded == first.runs[0].expanded
