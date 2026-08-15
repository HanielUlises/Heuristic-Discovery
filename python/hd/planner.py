"""Invocation of the C++ planner.

Python is deliberately kept outside the search loop: one subprocess call
evaluates a candidate heuristic on an entire benchmark suite, and the only
data crossing the boundary is the heuristic specification going in and a JSON
metrics document coming out.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

from .candidate import HeuristicCandidate
from .paths import planner_binary

#: Either a built-in baseline name or an explicit candidate.
HeuristicLike = Union[str, HeuristicCandidate]

SEARCH_ALGORITHMS = ("bfs", "gbfs", "astar")


@dataclass(frozen=True)
class RunMetrics:
    """Metrics of a single planner execution on a single instance."""

    instance: str
    instance_path: str
    solved: bool
    status: str
    solution_cost: Optional[float]
    solution_length: int
    expanded: int
    generated: int
    evaluated: int
    reopened: int
    max_depth: int
    peak_nodes: int
    runtime_seconds: float
    peak_memory_kb: int
    num_propositions: int
    num_actions: int
    num_goal_conditions: int
    plan: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, run: Mapping[str, object]) -> "RunMetrics":
        metrics = dict(run["metrics"])  # type: ignore[arg-type]
        return cls(
            instance=str(run["instance"]),
            instance_path=str(run["instance_path"]),
            solved=bool(metrics["solved"]),
            status=str(metrics["status"]),
            solution_cost=(
                None if metrics["solution_cost"] is None else float(metrics["solution_cost"])
            ),
            solution_length=int(metrics["solution_length"]),
            expanded=int(metrics["expanded"]),
            generated=int(metrics["generated"]),
            evaluated=int(metrics["evaluated"]),
            reopened=int(metrics["reopened"]),
            max_depth=int(metrics["max_depth"]),
            peak_nodes=int(metrics["peak_nodes"]),
            runtime_seconds=float(metrics["runtime_seconds"]),
            peak_memory_kb=int(metrics["peak_memory_kb"]),
            num_propositions=int(run["num_propositions"]),
            num_actions=int(run["num_actions"]),
            num_goal_conditions=int(run["num_goal_conditions"]),
            plan=list(run.get("plan", [])),  # type: ignore[arg-type]
        )

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PlannerRun:
    """One planner invocation over a set of instances."""

    search: str
    heuristic: Dict[str, object]
    limits: Dict[str, object]
    build: Dict[str, object]
    seed: int
    timestamp: str
    runs: List[RunMetrics]

    def by_instance(self) -> Dict[str, RunMetrics]:
        return {r.instance: r for r in self.runs}

    def to_dict(self) -> Dict[str, object]:
        return {
            "search": self.search,
            "heuristic": self.heuristic,
            "limits": self.limits,
            "build": self.build,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "runs": [r.to_dict() for r in self.runs],
        }


class PlannerError(RuntimeError):
    """Raised when the planner exits with a non-zero status."""


class Planner:
    """Thin, stateless wrapper around the ``hd_plan`` executable."""

    def __init__(self, binary: Optional[Path] = None) -> None:
        self.binary = Path(binary) if binary is not None else planner_binary()

    # --- introspection ----------------------------------------------------
    def features(self) -> List[Dict[str, object]]:
        """Feature registry reported by the binary itself."""
        return json.loads(self._check_output(["--list-features"]))

    def build_info(self) -> Dict[str, object]:
        return json.loads(self._check_output(["--build-info"]))

    # --- execution --------------------------------------------------------
    def run(
        self,
        instances: Sequence[Path | str],
        heuristic: HeuristicLike = "zero",
        search: str = "gbfs",
        max_expansions: int = 1_000_000,
        time_limit: float = 60.0,
        seed: int = 0,
        emit_plan: bool = False,
    ) -> PlannerRun:
        """Runs one (search, heuristic) configuration over ``instances``."""
        if search not in SEARCH_ALGORITHMS:
            raise ValueError(f"unknown search algorithm '{search}'")
        if not instances:
            raise ValueError("no instances given")

        spec = heuristic.to_spec() if isinstance(heuristic, HeuristicCandidate) else str(heuristic)
        argv: List[str] = [
            "--search",
            search,
            "--heuristic",
            spec,
            "--max-expansions",
            str(int(max_expansions)),
            "--time-limit",
            str(float(time_limit)),
            "--seed",
            str(int(seed)),
        ]
        if emit_plan:
            argv.append("--emit-plan")
        for instance in instances:
            argv += ["--instance", str(instance)]

        document = json.loads(self._check_output(argv))
        return PlannerRun(
            search=str(document["search"]),
            heuristic=dict(document["heuristic"]),
            limits=dict(document["limits"]),
            build=dict(document["build"]),
            seed=int(document["seed"]),
            timestamp=str(document["timestamp"]),
            runs=[RunMetrics.from_json(r) for r in document["runs"]],
        )

    # --- internals --------------------------------------------------------
    def _check_output(self, argv: Sequence[str]) -> str:
        completed = subprocess.run(
            [str(self.binary), *argv], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise PlannerError(
                f"{self.binary} exited with status {completed.returncode}\n"
                f"  argv: {' '.join(argv)}\n"
                f"  stderr: {completed.stderr.strip()}"
            )
        return completed.stdout
