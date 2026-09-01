"""Verification of candidate heuristics against exact goal distances.

A heuristic is admissible when it never overestimates the true cost to a goal.
Nothing in a search run reveals that: an inadmissible heuristic returns plans,
often quickly, and A* simply returns suboptimal ones. The property is decided
against h*, which on instances small enough to enumerate is computable exactly
by the ``hd_verify`` engine (see ``cpp/include/hd/oracle.hpp``).

This layer invokes that engine, parses its verdict, and aggregates it over a
suite. Three cautions belong with every number it produces:

* Enumeration is exponential in instance size, so only the small end of a
  suite can be checked. A clean report is evidence, not proof; a violation is
  a certificate, and the witness state it carries settles the matter.
* Instances too large to enumerate are reported as skipped, never as passing.
* Informedness (mean ``h / h*``) ranks admissible candidates without running
  a search. Above 1 it is not a quality measure at all, only a measure of how
  badly the candidate overestimates.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .candidate import HeuristicCandidate
from .paths import verifier_binary
from .planner import HeuristicLike

#: Ratios and excesses below this are floating-point noise, not violations.
DEFAULT_TOLERANCE = 1e-9

#: Enumeration ceiling. Blocksworld reaches it at about eight blocks.
DEFAULT_MAX_STATES = 200_000


@dataclass(frozen=True)
class StateSpaceSummary:
    """The enumerated state space of one instance."""

    status: str
    complete: bool
    states: int
    transitions: int
    goal_states: int
    dead_end_states: int
    max_h_star: float
    #: Optimal cost of the instance, or ``None`` when it is unsolvable.
    initial_h_star: Optional[float]
    runtime_seconds: float

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "StateSpaceSummary":
        return cls(
            status=str(data["status"]),
            complete=bool(data["complete"]),
            states=int(data["states"]),
            transitions=int(data["transitions"]),
            goal_states=int(data["goal_states"]),
            dead_end_states=int(data["dead_end_states"]),
            max_h_star=float(data["max_h_star"]),
            initial_h_star=(
                None if data["initial_h_star"] is None else float(data["initial_h_star"])
            ),
            runtime_seconds=float(data["runtime_seconds"]),
        )

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AdmissibilityWitness:
    """A reachable state at which the candidate overestimates."""

    h: float
    h_star: float
    excess: float
    ratio: Optional[float]
    state: Tuple[str, ...]

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "AdmissibilityWitness":
        ratio = data.get("ratio")
        return cls(
            h=float(data["h"]),
            h_star=float(data["h_star"]),
            excess=float(data["excess"]),
            ratio=None if ratio is None else float(ratio),
            state=tuple(str(p) for p in data["state"]),  # type: ignore[arg-type]
        )

    def to_dict(self) -> Dict[str, object]:
        return {**self.__dict__, "state": list(self.state)}

    def describe(self) -> str:
        return (
            f"h = {self.h:.4g} > h* = {self.h_star:.4g} at "
            "{" + ", ".join(self.state) + "}"
        )


@dataclass(frozen=True)
class ConsistencyWitness:
    """A transition along which the candidate falls by more than its cost."""

    action: str
    cost: float
    h_from: float
    h_to: float
    excess: float
    from_state: Tuple[str, ...]
    to_state: Tuple[str, ...]

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "ConsistencyWitness":
        return cls(
            action=str(data["action"]),
            cost=float(data["cost"]),
            h_from=float(data["h_from"]),
            h_to=float(data["h_to"]),
            excess=float(data["excess"]),
            from_state=tuple(str(p) for p in data["from_state"]),  # type: ignore[arg-type]
            to_state=tuple(str(p) for p in data["to_state"]),  # type: ignore[arg-type]
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            **self.__dict__,
            "from_state": list(self.from_state),
            "to_state": list(self.to_state),
        }

    def describe(self) -> str:
        return (
            f"h = {self.h_from:.4g} -> {self.h_to:.4g} across {self.action} "
            f"(cost {self.cost:.4g})"
        )


@dataclass(frozen=True)
class AdmissibilityReport:
    """The verdict on one heuristic on one instance."""

    admissible: bool
    consistent: bool
    goal_aware: bool
    non_negative: bool
    states_checked: int
    transitions_checked: int
    dead_end_states: int
    goal_states: int
    admissibility_violations: int
    consistency_violations: int
    max_excess: float
    max_ratio: float
    max_consistency_excess: float
    max_goal_value: float
    min_value: float
    max_value: float
    mean_informedness: float
    min_informedness: float
    mean_error: float
    admissibility_witnesses: List[AdmissibilityWitness] = field(default_factory=list)
    consistency_witnesses: List[ConsistencyWitness] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "AdmissibilityReport":
        def number(key: str) -> float:
            value = data[key]
            return float("inf") if value is None else float(value)  # type: ignore[arg-type]

        return cls(
            admissible=bool(data["admissible"]),
            consistent=bool(data["consistent"]),
            goal_aware=bool(data["goal_aware"]),
            non_negative=bool(data["non_negative"]),
            states_checked=int(data["states_checked"]),  # type: ignore[arg-type]
            transitions_checked=int(data["transitions_checked"]),  # type: ignore[arg-type]
            dead_end_states=int(data["dead_end_states"]),  # type: ignore[arg-type]
            goal_states=int(data["goal_states"]),  # type: ignore[arg-type]
            admissibility_violations=int(data["admissibility_violations"]),  # type: ignore[arg-type]
            consistency_violations=int(data["consistency_violations"]),  # type: ignore[arg-type]
            max_excess=number("max_excess"),
            max_ratio=number("max_ratio"),
            max_consistency_excess=number("max_consistency_excess"),
            max_goal_value=number("max_goal_value"),
            min_value=number("min_value"),
            max_value=number("max_value"),
            mean_informedness=number("mean_informedness"),
            min_informedness=number("min_informedness"),
            mean_error=number("mean_error"),
            admissibility_witnesses=[
                AdmissibilityWitness.from_json(w)
                for w in data.get("admissibility_witnesses", [])  # type: ignore[union-attr]
            ],
            consistency_witnesses=[
                ConsistencyWitness.from_json(w)
                for w in data.get("consistency_witnesses", [])  # type: ignore[union-attr]
            ],
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            **{
                k: v
                for k, v in self.__dict__.items()
                if k not in ("admissibility_witnesses", "consistency_witnesses")
            },
            "admissibility_witnesses": [w.to_dict() for w in self.admissibility_witnesses],
            "consistency_witnesses": [w.to_dict() for w in self.consistency_witnesses],
        }


@dataclass(frozen=True)
class InstanceVerification:
    """Every heuristic checked on one instance, with that instance's space."""

    instance: str
    instance_path: str
    state_space: StateSpaceSummary
    reports: Dict[str, AdmissibilityReport] = field(default_factory=dict)

    @property
    def checked(self) -> bool:
        """Whether a verdict was produced; false when the space was truncated."""
        return self.state_space.complete

    def to_dict(self) -> Dict[str, object]:
        return {
            "instance": self.instance,
            "instance_path": self.instance_path,
            "state_space": self.state_space.to_dict(),
            "reports": {k: v.to_dict() for k, v in self.reports.items()},
        }


@dataclass(frozen=True)
class HeuristicSummary:
    """One heuristic aggregated over the instances that could be enumerated."""

    heuristic: str
    instances_checked: int
    instances_skipped: int
    admissible: bool
    consistent: bool
    violating_instances: int
    total_violations: int
    max_excess: float
    max_ratio: float
    mean_informedness: float
    witness: Optional[AdmissibilityWitness] = None

    def verdict(self) -> str:
        """A one-line reading of the verdict.

        "admissible" here always means "not falsified on the instances that
        were enumerable", which is the strongest statement this method can
        make; only "inadmissible" is conclusive.
        """
        if self.instances_checked == 0:
            return "unchecked"
        if not self.admissible:
            return "inadmissible"
        return "admissible, consistent" if self.consistent else "admissible, inconsistent"

    def to_dict(self) -> Dict[str, object]:
        return {
            **{k: v for k, v in self.__dict__.items() if k != "witness"},
            "verdict": self.verdict(),
            "witness": None if self.witness is None else self.witness.to_dict(),
        }


@dataclass(frozen=True)
class VerificationRun:
    """One invocation of ``hd_verify`` over a set of instances."""

    timestamp: str
    limits: Dict[str, object]
    build: Dict[str, object]
    heuristics: List[str]
    instances: List[InstanceVerification]

    def skipped(self) -> List[InstanceVerification]:
        """Instances whose state space was too large to enumerate."""
        return [i for i in self.instances if not i.checked]

    def summary(self, heuristic: str) -> HeuristicSummary:
        """Aggregates one heuristic over every instance that was checked."""
        if heuristic not in self.heuristics:
            raise KeyError(f"'{heuristic}' was not verified in this run")
        reports = [i.reports[heuristic] for i in self.instances if i.checked]
        skipped = len(self.instances) - len(reports)
        if not reports:
            return HeuristicSummary(
                heuristic=heuristic,
                instances_checked=0,
                instances_skipped=skipped,
                admissible=False,
                consistent=False,
                violating_instances=0,
                total_violations=0,
                max_excess=0.0,
                max_ratio=0.0,
                mean_informedness=0.0,
            )

        violating = [r for r in reports if not r.admissible]
        witnesses = [w for r in violating for w in r.admissibility_witnesses]
        return HeuristicSummary(
            heuristic=heuristic,
            instances_checked=len(reports),
            instances_skipped=skipped,
            admissible=not violating,
            consistent=all(r.consistent for r in reports),
            violating_instances=len(violating),
            total_violations=sum(r.admissibility_violations for r in reports),
            max_excess=max(r.max_excess for r in reports),
            max_ratio=max(r.max_ratio for r in reports),
            mean_informedness=mean(r.mean_informedness for r in reports),
            witness=max(witnesses, key=lambda w: w.excess) if witnesses else None,
        )

    def summaries(self) -> List[HeuristicSummary]:
        return [self.summary(h) for h in self.heuristics]

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "limits": dict(self.limits),
            "build": dict(self.build),
            "heuristics": list(self.heuristics),
            "instances": [i.to_dict() for i in self.instances],
            "summaries": [s.to_dict() for s in self.summaries()],
        }


class VerificationError(RuntimeError):
    """Raised when the verifier exits with a non-zero status."""


def heuristic_label(heuristic: HeuristicLike) -> str:
    """The key a heuristic's reports are filed under."""
    return heuristic.name if isinstance(heuristic, HeuristicCandidate) else str(heuristic)


class Verifier:
    """Thin, stateless wrapper around the ``hd_verify`` executable."""

    def __init__(self, binary: Optional[Path] = None) -> None:
        self.binary = Path(binary) if binary is not None else verifier_binary()

    def verify(
        self,
        instances: Sequence[Path | str],
        heuristics: Sequence[HeuristicLike] = ("relaxed_layers",),
        max_states: int = DEFAULT_MAX_STATES,
        time_limit: float = 0.0,
        witnesses: int = 3,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> VerificationRun:
        """Checks every heuristic on every instance small enough to enumerate.

        The state space of an instance is enumerated once and shared by all
        heuristics, so the cost of checking a set of candidates is dominated
        by the instances rather than by their number.
        """
        if not instances:
            raise ValueError("no instances given")
        if not heuristics:
            raise ValueError("no heuristics given")

        labels = [heuristic_label(h) for h in heuristics]
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise ValueError(f"duplicate heuristic label(s): {', '.join(sorted(duplicates))}")

        argv: List[str] = [
            "--max-states",
            str(int(max_states)),
            "--time-limit",
            str(float(time_limit)),
            "--witnesses",
            str(int(witnesses)),
            "--tolerance",
            repr(float(tolerance)),
        ]
        for heuristic in heuristics:
            spec = (
                heuristic.to_spec()
                if isinstance(heuristic, HeuristicCandidate)
                else str(heuristic)
            )
            argv += ["--heuristic", spec]
        for instance in instances:
            argv += ["--instance", str(instance)]

        document = json.loads(self._check_output(argv))
        return VerificationRun(
            timestamp=str(document["timestamp"]),
            limits=dict(document["limits"]),
            build=dict(document["build"]),
            heuristics=labels,
            instances=[_instance_from_json(run, labels) for run in document["runs"]],
        )

    def _check_output(self, argv: Sequence[str]) -> str:
        completed = subprocess.run(
            [str(self.binary), *argv], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise VerificationError(
                f"{self.binary} exited with status {completed.returncode}\n"
                f"  argv: {' '.join(argv)}\n"
                f"  stderr: {completed.stderr.strip()}"
            )
        return completed.stdout


def _instance_from_json(run: Mapping[str, object], labels: Sequence[str]) -> InstanceVerification:
    # Reports arrive in the order the heuristics were requested, and are absent
    # entirely when the state space could not be enumerated.
    entries = list(run["heuristics"])  # type: ignore[arg-type]
    reports = {
        labels[i]: AdmissibilityReport.from_json(entry["report"]) for i, entry in enumerate(entries)
    }
    return InstanceVerification(
        instance=str(run["instance"]),
        instance_path=str(run["instance_path"]),
        state_space=StateSpaceSummary.from_json(run["state_space"]),  # type: ignore[arg-type]
        reports=reports,
    )


def summary_table(run: VerificationRun) -> str:
    """A fixed-width table of verdicts, one row per heuristic."""
    header = (
        f"{'heuristic':<24}{'verdict':>26}{'violations':>12}"
        f"{'max h-h*':>11}{'informedness':>14}"
    )
    lines = [header, "-" * len(header)]
    for s in run.summaries():
        informedness = "-" if s.instances_checked == 0 else f"{s.mean_informedness:.3f}"
        lines.append(
            f"{s.heuristic:<24}{s.verdict():>26}{s.total_violations:>12}"
            f"{s.max_excess:>11.3f}{informedness:>14}"
        )
    skipped = run.skipped()
    if skipped:
        lines.append("")
        lines.append(
            f"{len(skipped)} instance(s) not enumerable within the ceiling and left unchecked: "
            + ", ".join(i.instance for i in skipped[:5])
            + (", ..." if len(skipped) > 5 else "")
        )
    return "\n".join(lines)
