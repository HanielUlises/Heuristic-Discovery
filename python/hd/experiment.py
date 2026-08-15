"""Experiment execution and provenance.

Every experiment writes a single self-describing record: the seed, the planner
build, the benchmark suite, the objective weights, the search space, the full
optimisation trajectory and the resulting metrics. The record is sufficient to
re-run the experiment and to check a reported number long after the fact.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .benchmark import BenchmarkSuite, BenchmarkSummary, summarise
from .candidate import HeuristicCandidate
from .objective import Objective, ObjectiveValue, ObjectiveWeights
from .optimize import OptimizationResult, SearchSpace
from .paths import RESULTS_DIR, ROOT, relative_to_root
from .planner import HeuristicLike, Planner, PlannerRun


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:  # pragma: no cover - git absent
        return "unknown"


def environment_info(planner: Planner) -> Dict[str, object]:
    """Provenance of the software that produced a measurement."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_revision": git_revision(),
        "planner_binary": str(planner.binary),
        "planner_build": planner.build_info(),
    }


@dataclass
class Evaluation:
    """A candidate, its planner run, and its score."""

    candidate: HeuristicCandidate
    run: PlannerRun
    objective: ObjectiveValue
    summary: BenchmarkSummary

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "objective": self.objective.to_dict(),
            "summary": self.summary.to_dict(),
            "runs": [r.to_dict() for r in self.run.runs],
        }


class Experiment:
    """Binds a planner, a suite and an objective into a scoring function.

    The instance is callable on a :class:`HeuristicCandidate`, which is exactly
    the interface the optimisers consume. Identical candidates are evaluated
    once: the planner is deterministic, so caching changes nothing except cost.
    """

    def __init__(
        self,
        suite: BenchmarkSuite,
        planner: Optional[Planner] = None,
        weights: ObjectiveWeights = ObjectiveWeights(),
        search: str = "gbfs",
        max_expansions: int = 200_000,
        time_limit: float = 30.0,
        seed: int = 0,
        reference_heuristic: HeuristicLike = "goal_count",
    ) -> None:
        self.suite = suite
        self.planner = planner or Planner()
        self.search = search
        self.max_expansions = max_expansions
        self.time_limit = time_limit
        self.seed = seed
        self.weights = weights

        self.reference_heuristic = reference_heuristic
        self.reference_run = self.run(reference_heuristic)
        self.objective = Objective.from_reference_run(self.reference_run, weights)
        self._cache: Dict[str, Evaluation] = {}

    # --- planner access ---------------------------------------------------
    def run(self, heuristic: HeuristicLike, **overrides: object) -> PlannerRun:
        return self.planner.run(
            self.suite.instances,
            heuristic=heuristic,
            search=str(overrides.get("search", self.search)),
            max_expansions=int(overrides.get("max_expansions", self.max_expansions)),
            time_limit=float(overrides.get("time_limit", self.time_limit)),
            seed=int(overrides.get("seed", self.seed)),
        )

    def evaluate(self, heuristic: HeuristicLike) -> Evaluation:
        """Runs the suite under ``heuristic`` and scores it."""
        key = (
            heuristic.to_spec() if isinstance(heuristic, HeuristicCandidate) else str(heuristic)
        )
        if key not in self._cache:
            run = self.run(heuristic)
            candidate = (
                heuristic
                if isinstance(heuristic, HeuristicCandidate)
                else HeuristicCandidate({}, name=str(heuristic))
            )
            self._cache[key] = Evaluation(
                candidate=candidate,
                run=run,
                objective=self.objective(run),
                summary=summarise(run.runs),
            )
        return self._cache[key]

    def __call__(self, candidate: HeuristicCandidate) -> ObjectiveValue:
        return self.evaluate(candidate).objective

    def scorer(self) -> Callable[[HeuristicCandidate], ObjectiveValue]:
        return self.__call__

    def config(self) -> Dict[str, object]:
        return {
            "suite": self.suite.name,
            "num_instances": len(self.suite),
            "search": self.search,
            "max_expansions": self.max_expansions,
            "time_limit": self.time_limit,
            "seed": self.seed,
            "objective_weights": self.weights.to_dict(),
            "reference_heuristic": (
                self.reference_heuristic.to_spec()
                if isinstance(self.reference_heuristic, HeuristicCandidate)
                else str(self.reference_heuristic)
            ),
        }


@dataclass
class ExperimentRecord:
    """The serialisable record of one experiment."""

    name: str
    seed: int
    config: Dict[str, object]
    environment: Dict[str, object]
    instances: List[str]
    search_space: Optional[Dict[str, object]] = None
    optimization: Optional[Dict[str, object]] = None
    evaluations: Dict[str, object] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "hd.experiment/1",
            "name": self.name,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "config": self.config,
            "environment": self.environment,
            "instances": self.instances,
            "search_space": self.search_space,
            "optimization": self.optimization,
            "evaluations": self.evaluations,
        }

    def save(self, directory: Optional[Path] = None) -> Path:
        directory = Path(directory or RESULTS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp.replace(":", "").replace("-", "")
        path = directory / f"{self.name}-{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> Dict[str, object]:
        return json.loads(Path(path).read_text(encoding="utf-8"))


def build_record(
    name: str,
    experiment: Experiment,
    evaluations: Dict[str, Evaluation],
    optimization: Optional[OptimizationResult] = None,
    space: Optional[SearchSpace] = None,
) -> ExperimentRecord:
    """Assembles the record of a finished experiment."""
    return ExperimentRecord(
        name=name,
        seed=experiment.seed,
        config=experiment.config(),
        environment=environment_info(experiment.planner),
        instances=[relative_to_root(p) for p in experiment.suite.instances],
        search_space=space.to_dict() if space is not None else None,
        optimization=optimization.to_dict() if optimization is not None else None,
        evaluations={k: v.to_dict() for k, v in evaluations.items()},
    )
