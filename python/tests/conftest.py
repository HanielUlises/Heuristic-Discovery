"""Shared fixtures: synthetic metrics, and access to the compiled planner."""

from typing import Optional

import pytest

from hd.paths import ROOT
from hd.planner import Planner, PlannerRun, RunMetrics


def make_metrics(
    instance: str = "i0",
    expanded: int = 100,
    runtime: float = 0.1,
    cost: Optional[float] = 10.0,
    solved: bool = True,
) -> RunMetrics:
    """A metrics record with plausible values, for testing aggregation."""
    return RunMetrics(
        instance=instance,
        instance_path=f"instances/{instance}.task",
        solved=solved,
        status="solved" if solved else "expansion_limit",
        solution_cost=cost if solved else None,
        solution_length=int(cost) if solved and cost else 0,
        expanded=expanded,
        generated=expanded * 3,
        evaluated=expanded * 3,
        reopened=0,
        max_depth=int(cost) if solved and cost else 0,
        peak_nodes=expanded * 3,
        runtime_seconds=runtime,
        peak_memory_kb=1024,
        num_propositions=16,
        num_actions=12,
        num_goal_conditions=3,
    )


def make_run(*metrics: RunMetrics, search: str = "gbfs") -> PlannerRun:
    return PlannerRun(
        search=search,
        heuristic={"kind": "linear", "weights": {}},
        limits={"max_expansions": 1000, "time_limit_seconds": 10.0},
        build={"version": "test"},
        seed=0,
        timestamp="2024-01-01T00:00:00Z",
        runs=list(metrics),
    )


@pytest.fixture(scope="session")
def planner() -> Planner:
    """The compiled planner, skipping the test when it has not been built."""
    try:
        return Planner()
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="session")
def benchmark_manifest():
    """Path of the generated benchmark manifest, skipping when absent."""
    path = ROOT / "benchmarks" / "blocksworld.json"
    if not path.exists():
        pytest.skip("run scripts/generate_benchmarks.py first")
    return path
