"""Heuristic Discovery: the research layer around the C++ planning engine.

The package provides the outer loop of heuristic discovery — candidate
generation, benchmark execution, objective evaluation and experiment
recording — while all state-space search happens inside the compiled planner.
"""

from .benchmark import BenchmarkSuite, BenchmarkSummary, comparison_table, summarise
from .candidate import BASELINE_HEURISTICS, FEATURE_NAMES, HeuristicCandidate
from .experiment import Evaluation, Experiment, ExperimentRecord, build_record
from .objective import Objective, ObjectiveValue, ObjectiveWeights
from .optimize import (
    OptimizationResult,
    SearchSpace,
    default_search_space,
    grid_search,
    local_search,
    random_search,
)
from .planner import Planner, PlannerRun, RunMetrics

__version__ = "0.1.0"

__all__ = [
    "BASELINE_HEURISTICS",
    "BenchmarkSuite",
    "BenchmarkSummary",
    "Evaluation",
    "Experiment",
    "ExperimentRecord",
    "FEATURE_NAMES",
    "HeuristicCandidate",
    "Objective",
    "ObjectiveValue",
    "ObjectiveWeights",
    "OptimizationResult",
    "Planner",
    "PlannerRun",
    "RunMetrics",
    "SearchSpace",
    "build_record",
    "comparison_table",
    "default_search_space",
    "grid_search",
    "local_search",
    "random_search",
    "summarise",
]
