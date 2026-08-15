"""Benchmark suites, aggregation and the domain generator."""

import pytest
from conftest import make_metrics, make_run

from hd.benchmark import BenchmarkSuite, comparison_table, summarise
from hd.domains import blocksworld


def write_instances(tmp_path, count=3):
    paths = []
    for i in range(count):
        path = tmp_path / f"i{i}.task"
        path.write_text("propositions 2\ninit 0\ngoal 1\n")
        paths.append(path)
    return paths


def test_suite_manifest_round_trip(tmp_path):
    suite = BenchmarkSuite(
        name="toy", instances=write_instances(tmp_path), description="d", metadata={"seed": 1}
    )
    manifest = suite.save(tmp_path / "toy.json")
    restored = BenchmarkSuite.load(manifest)
    assert restored.name == "toy"
    assert len(restored) == 3
    assert restored.metadata["seed"] == 1


def test_missing_instances_are_reported_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing instance"):
        BenchmarkSuite(name="toy", instances=[tmp_path / "absent.task"])


def test_subset_keeps_the_leading_instances(tmp_path):
    suite = BenchmarkSuite(name="toy", instances=write_instances(tmp_path, 4))
    assert len(suite.subset(2)) == 2


def test_summary_aggregates_totals_and_coverage():
    summary = summarise(
        [
            make_metrics("a", expanded=100, runtime=0.1, cost=10.0),
            make_metrics("b", expanded=300, runtime=0.3, cost=None, solved=False),
        ]
    )
    assert summary.num_instances == 2
    assert summary.num_solved == 1
    assert summary.coverage == pytest.approx(0.5)
    assert summary.total_expanded == 400
    assert summary.mean_expanded == pytest.approx(200.0)
    assert summary.total_cost == pytest.approx(10.0)  # unsolved contributes no cost
    assert summary.mean_cost == pytest.approx(10.0)


def test_summary_of_an_empty_run_is_neutral():
    summary = summarise([])
    assert summary.num_instances == 0
    assert summary.coverage == 0.0


def test_comparison_table_lists_instances_and_totals():
    runs = {
        "baseline": make_run(make_metrics("a", expanded=100), make_metrics("b", expanded=200)),
        "learned": make_run(make_metrics("a", expanded=10), make_metrics("b", expanded=20)),
    }
    table = comparison_table(runs)
    assert "baseline" in table and "learned" in table
    assert "300" in table and "30" in table


# --- domain generator -----------------------------------------------------


def test_generated_task_has_the_expected_proposition_count():
    task = blocksworld.generate(num_blocks=4, seed=1)
    # n(n-1) on + n ontable + n clear + n holding + handempty
    assert task.num_propositions == 4 * 3 + 3 * 4 + 1
    assert len(task.actions) == 2 * 4 + 2 * 4 * 3


def test_generation_is_a_function_of_the_seed():
    first = blocksworld.generate(num_blocks=5, seed=11).to_task_text("x")
    second = blocksworld.generate(num_blocks=5, seed=11).to_task_text("x")
    assert first == second
    assert first != blocksworld.generate(num_blocks=5, seed=12).to_task_text("x")


def test_the_goal_is_a_non_trivial_partial_state():
    task = blocksworld.generate(num_blocks=5, seed=3)
    assert 0 < len(task.goal) <= task.num_propositions
    assert not set(task.goal).issubset(set(task.initial))


def test_task_text_is_parseable_by_the_planner_format():
    text = blocksworld.generate(num_blocks=3, seed=2).to_task_text("blocks-3")
    assert text.startswith("name blocks-3\npropositions 16\n")  # 3*2 on + 3*3 + handempty
    assert "\ninit " in text and "\ngoal " in text
    assert text.count("\naction ") == text.count("\nend")


def test_too_few_blocks_is_rejected():
    with pytest.raises(ValueError):
        blocksworld.generate(num_blocks=1, seed=0)


def test_suite_specification_is_stable_and_sized_correctly():
    spec = blocksworld.suite_specification(block_counts=(3, 4), instances_per_size=3)
    assert len(spec) == 6
    assert spec == blocksworld.suite_specification(block_counts=(3, 4), instances_per_size=3)
    assert len({name for name, _, _ in spec}) == 6
    assert len({seed for _, _, seed in spec}) == 6
