"""Benchmark suites and their aggregate statistics.

A suite is a named, ordered list of instances stored as a JSON manifest, so
that an experiment records exactly which problems were solved and a later run
can reproduce the same set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

from .paths import BENCHMARK_DIR, ROOT, relative_to_root
from .planner import PlannerRun, RunMetrics


@dataclass(frozen=True)
class BenchmarkSuite:
    """An ordered collection of instance files."""

    name: str
    instances: List[Path]
    description: str = ""
    metadata: Dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        missing = [p for p in self.instances if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                "missing instance file(s): "
                + ", ".join(str(p) for p in missing[:3])
                + "\nGenerate the benchmark suite with: python scripts/generate_benchmarks.py"
            )

    def __len__(self) -> int:
        return len(self.instances)

    @classmethod
    def load(cls, path: Path | str) -> "BenchmarkSuite":
        """Loads a suite manifest; bare names resolve inside ``benchmarks/``."""
        path = Path(path)
        if not path.exists() and path.suffix == "":
            path = BENCHMARK_DIR / f"{path.name}.json"
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(data["name"]),
            instances=[_resolve(p) for p in data["instances"]],
            description=str(data.get("description", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema": "hd.benchmark_suite/1",
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "instances": [relative_to_root(p) for p in self.instances],
        }
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    def subset(self, limit: int) -> "BenchmarkSuite":
        return BenchmarkSuite(
            name=f"{self.name}[:{limit}]",
            instances=self.instances[:limit],
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate statistics of one configuration over one suite."""

    num_instances: int
    num_solved: int
    total_expanded: int
    total_generated: int
    total_runtime: float
    total_cost: float
    mean_expanded: float
    mean_runtime: float
    mean_cost: float
    peak_memory_kb: int

    @property
    def coverage(self) -> float:
        return self.num_solved / self.num_instances if self.num_instances else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {**self.__dict__, "coverage": self.coverage}


def summarise(runs: Sequence[RunMetrics]) -> BenchmarkSummary:
    """Aggregates per-instance metrics.

    Cost statistics are taken over solved instances only, since an unsolved
    instance has no cost; coverage reports the rest.
    """
    solved = [r for r in runs if r.solved]
    costs = [r.solution_cost or 0.0 for r in solved]
    return BenchmarkSummary(
        num_instances=len(runs),
        num_solved=len(solved),
        total_expanded=sum(r.expanded for r in runs),
        total_generated=sum(r.generated for r in runs),
        total_runtime=sum(r.runtime_seconds for r in runs),
        total_cost=sum(costs),
        mean_expanded=mean([r.expanded for r in runs]) if runs else 0.0,
        mean_runtime=mean([r.runtime_seconds for r in runs]) if runs else 0.0,
        mean_cost=mean(costs) if costs else 0.0,
        peak_memory_kb=max((r.peak_memory_kb for r in runs), default=0),
    )


def summarise_run(run: PlannerRun) -> BenchmarkSummary:
    return summarise(run.runs)


def comparison_table(
    named_runs: Dict[str, PlannerRun], instance_names: Optional[Sequence[str]] = None
) -> str:
    """Renders expansions per instance for several configurations."""
    columns = list(named_runs)
    tables = {name: run.by_instance() for name, run in named_runs.items()}
    instances = list(instance_names or next(iter(tables.values())).keys())

    width = max([len(i) for i in instances] + [8])
    header = "instance".ljust(width) + "".join(f"{c:>14}" for c in columns)
    lines = [header, "-" * len(header)]
    for instance in instances:
        row = instance.ljust(width)
        for column in columns:
            metrics = tables[column].get(instance)
            row += f"{metrics.expanded:>14}" if metrics and metrics.solved else f"{'-':>14}"
        lines.append(row)
    total = "total".ljust(width)
    for column in columns:
        total += f"{sum(m.expanded for m in tables[column].values()):>14}"
    lines += ["-" * len(header), total]
    return "\n".join(lines)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate
