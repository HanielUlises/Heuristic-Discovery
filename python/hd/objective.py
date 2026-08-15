"""The objective minimised by heuristic discovery.

For a candidate theta evaluated on a suite of instances I, the objective is

    J(theta) = alpha * E[ expanded(i, theta)  / expanded(i, ref)  ]
             + beta  * E[ runtime(i, theta)   / runtime(i, ref)   ]
             + gamma * E[ cost(i, theta)      / cost(i, ref)      ]
             + delta * (1 - coverage(theta))

with the expectation taken over instances i in I. Every term is normalised by
a fixed reference configuration, so instances of very different sizes
contribute comparably and the objective is scale-free: J = 1 means "as good as
the reference", J < 1 means better. The last term prices failure to solve an
instance, which is otherwise invisible to ratios of search effort.

Lower is better throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Mapping, Optional, Sequence

from .planner import PlannerRun, RunMetrics

#: Guards the ratios against division by zero on instances solved instantly.
EPSILON = 1e-6


@dataclass(frozen=True)
class ObjectiveWeights:
    """Relative importance of the objective terms."""

    expanded: float = 1.0  # alpha
    runtime: float = 0.0  # beta
    cost: float = 0.0  # gamma
    unsolved: float = 10.0  # delta

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> "ObjectiveWeights":
        return cls(**{k: float(v) for k, v in data.items()})


@dataclass(frozen=True)
class ObjectiveValue:
    """The objective value together with its unweighted components."""

    value: float
    components: Dict[str, float] = field(default_factory=dict)
    coverage: float = 0.0

    def __float__(self) -> float:
        return self.value

    def to_dict(self) -> Dict[str, object]:
        return {"value": self.value, "components": dict(self.components), "coverage": self.coverage}


class Objective:
    """Scores a planner run against a fixed reference run.

    The reference is normally a baseline configuration (for example greedy
    best-first search with the goal-count heuristic) evaluated once on the same
    suite. Holding it fixed for the whole experiment keeps candidate scores
    comparable across the optimisation trajectory.
    """

    def __init__(
        self,
        reference: Optional[Mapping[str, RunMetrics]] = None,
        weights: ObjectiveWeights = ObjectiveWeights(),
    ) -> None:
        self.reference = dict(reference or {})
        self.weights = weights

    @classmethod
    def from_reference_run(
        cls, run: PlannerRun, weights: ObjectiveWeights = ObjectiveWeights()
    ) -> "Objective":
        return cls(reference=run.by_instance(), weights=weights)

    def __call__(self, run: PlannerRun) -> ObjectiveValue:
        return self.evaluate(run.runs)

    def evaluate(self, runs: Sequence[RunMetrics]) -> ObjectiveValue:
        if not runs:
            raise ValueError("cannot score an empty run")

        expanded_ratios = []
        runtime_ratios = []
        cost_ratios = []
        for metrics in runs:
            reference = self.reference.get(metrics.instance)
            expanded_ratios.append(_ratio(metrics.expanded, _ref(reference, "expanded")))
            runtime_ratios.append(
                _ratio(metrics.runtime_seconds, _ref(reference, "runtime_seconds"))
            )
            if metrics.solved and metrics.solution_cost is not None:
                cost_ratios.append(_ratio(metrics.solution_cost, _ref(reference, "solution_cost")))

        coverage = sum(1 for m in runs if m.solved) / len(runs)
        components = {
            "expanded": mean(expanded_ratios),
            "runtime": mean(runtime_ratios),
            "cost": mean(cost_ratios) if cost_ratios else 0.0,
            "unsolved": 1.0 - coverage,
        }
        weights = self.weights
        value = (
            weights.expanded * components["expanded"]
            + weights.runtime * components["runtime"]
            + weights.cost * components["cost"]
            + weights.unsolved * components["unsolved"]
        )
        return ObjectiveValue(value=value, components=components, coverage=coverage)


def _ref(metrics: Optional[RunMetrics], attribute: str) -> Optional[float]:
    if metrics is None:
        return None
    value = getattr(metrics, attribute)
    return None if value is None else float(value)


def _ratio(value: float, reference: Optional[float]) -> float:
    """Normalises ``value`` by ``reference``; unnormalised when absent."""
    if reference is None:
        return float(value)
    return (float(value) + EPSILON) / (reference + EPSILON)
