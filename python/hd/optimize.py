"""Phase I optimisers over the weight vector theta.

Only derivative-free search over an explicit, bounded weight space is provided:
random search, a randomised local search around the incumbent, and an exhaustive
grid. These are weak optimisers by design. They establish the loop

    candidate -> planner -> metrics -> objective -> candidate

against which the learning-based methods of the next phase must be measured;
a learned method that cannot beat random search on this benchmark has not
demonstrated anything.

Every optimiser is driven by an explicit ``random.Random`` seed and touches no
global random state, so a run is reproducible from its recorded seed alone.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .candidate import FEATURE_NAMES, HeuristicCandidate
from .objective import ObjectiveValue

#: Scores a candidate; lower is better.
EvaluateFn = Callable[[HeuristicCandidate], ObjectiveValue]


@dataclass(frozen=True)
class SearchSpace:
    """Box constraints on the weight of each participating feature."""

    bounds: Dict[str, Tuple[float, float]]

    def __post_init__(self) -> None:
        unknown = sorted(set(self.bounds) - set(FEATURE_NAMES))
        if unknown:
            raise ValueError(f"unknown feature(s) in the search space: {', '.join(unknown)}")
        for feature, (low, high) in self.bounds.items():
            if low > high:
                raise ValueError(f"empty interval for '{feature}': [{low}, {high}]")

    @property
    def features(self) -> List[str]:
        return list(self.bounds)

    def sample(self, rng: random.Random) -> Dict[str, float]:
        return {f: rng.uniform(low, high) for f, (low, high) in self.bounds.items()}

    def clip(self, weights: Mapping[str, float]) -> Dict[str, float]:
        clipped = {}
        for feature, value in weights.items():
            low, high = self.bounds[feature]
            clipped[feature] = min(max(value, low), high)
        return clipped

    def width(self, feature: str) -> float:
        low, high = self.bounds[feature]
        return high - low

    def to_dict(self) -> Dict[str, List[float]]:
        return {f: [low, high] for f, (low, high) in self.bounds.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[float]]) -> "SearchSpace":
        return cls({str(k): (float(v[0]), float(v[1])) for k, v in data.items()})


@dataclass(frozen=True)
class Trial:
    """One evaluated candidate."""

    iteration: int
    candidate: HeuristicCandidate
    objective: ObjectiveValue
    accepted: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "iteration": self.iteration,
            "candidate": self.candidate.to_dict(),
            "objective": self.objective.to_dict(),
            "accepted": self.accepted,
        }


@dataclass
class OptimizationResult:
    """Outcome of one optimisation run, including the full trajectory."""

    optimizer: str
    seed: int
    best_candidate: HeuristicCandidate
    best_objective: ObjectiveValue
    history: List[Trial] = field(default_factory=list)

    @property
    def num_evaluations(self) -> int:
        return len(self.history)

    def to_dict(self) -> Dict[str, object]:
        return {
            "optimizer": self.optimizer,
            "seed": self.seed,
            "num_evaluations": self.num_evaluations,
            "best_candidate": self.best_candidate.to_dict(),
            "best_objective": self.best_objective.to_dict(),
            "history": [t.to_dict() for t in self.history],
        }


def _record(
    history: List[Trial],
    iteration: int,
    candidate: HeuristicCandidate,
    value: ObjectiveValue,
    best: Optional[Tuple[HeuristicCandidate, ObjectiveValue]],
    on_trial: Optional[Callable[[Trial], None]],
) -> Tuple[HeuristicCandidate, ObjectiveValue]:
    improved = best is None or value.value < best[1].value
    history.append(Trial(iteration, candidate, value, accepted=improved))
    if on_trial is not None:
        on_trial(history[-1])
    return (candidate, value) if improved else best  # type: ignore[return-value]


def random_search(
    space: SearchSpace,
    evaluate: EvaluateFn,
    iterations: int = 32,
    seed: int = 0,
    initial: Optional[HeuristicCandidate] = None,
    on_trial: Optional[Callable[[Trial], None]] = None,
) -> OptimizationResult:
    """Samples weight vectors uniformly from the box.

    The reference optimiser: unbiased, embarrassingly parallel in principle,
    and the baseline every later method is compared against.
    """
    rng = random.Random(seed)
    history: List[Trial] = []
    best: Optional[Tuple[HeuristicCandidate, ObjectiveValue]] = None

    if initial is not None:
        best = _record(history, 0, initial, evaluate(initial), best, on_trial)

    for iteration in range(len(history), len(history) + iterations):
        candidate = HeuristicCandidate(space.sample(rng), name=f"random-{iteration:03d}")
        best = _record(history, iteration, candidate, evaluate(candidate), best, on_trial)

    assert best is not None
    return OptimizationResult("random_search", seed, best[0], best[1], history)


def local_search(
    space: SearchSpace,
    evaluate: EvaluateFn,
    iterations: int = 32,
    seed: int = 0,
    initial: Optional[HeuristicCandidate] = None,
    step_fraction: float = 0.25,
    decay: float = 0.97,
    on_trial: Optional[Callable[[Trial], None]] = None,
) -> OptimizationResult:
    """Gaussian perturbation of the incumbent with a shrinking step size.

    A hill climber over a noisy, non-differentiable objective: cheap, and
    usually a stronger baseline than uniform sampling once the incumbent is
    reasonable.
    """
    rng = random.Random(seed)
    history: List[Trial] = []

    incumbent = initial or HeuristicCandidate(space.sample(rng), name="local-000")
    best = _record(history, 0, incumbent, evaluate(incumbent), None, on_trial)

    scale = step_fraction
    for iteration in range(1, iterations + 1):
        weights = {
            feature: best[0].weights.get(feature, 0.0)
            + rng.gauss(0.0, max(scale * space.width(feature), 1e-9))
            for feature in space.features
        }
        candidate = HeuristicCandidate(space.clip(weights), name=f"local-{iteration:03d}")
        best = _record(history, iteration, candidate, evaluate(candidate), best, on_trial)
        scale *= decay

    return OptimizationResult("local_search", seed, best[0], best[1], history)


def grid_search(
    space: SearchSpace,
    evaluate: EvaluateFn,
    points_per_axis: int = 3,
    seed: int = 0,
    on_trial: Optional[Callable[[Trial], None]] = None,
) -> OptimizationResult:
    """Exhaustive grid over the box.

    Included for completeness on low-dimensional spaces; the number of
    evaluations grows as ``points_per_axis ** len(space.features)``.
    """
    if points_per_axis < 1:
        raise ValueError("points_per_axis must be positive")
    features = space.features
    axes = []
    for feature in features:
        low, high = space.bounds[feature]
        if points_per_axis == 1:
            axes.append([0.5 * (low + high)])
        else:
            step = (high - low) / (points_per_axis - 1)
            axes.append([low + step * k for k in range(points_per_axis)])

    history: List[Trial] = []
    best: Optional[Tuple[HeuristicCandidate, ObjectiveValue]] = None
    for iteration, point in enumerate(itertools.product(*axes)):
        candidate = HeuristicCandidate(dict(zip(features, point)), name=f"grid-{iteration:03d}")
        best = _record(history, iteration, candidate, evaluate(candidate), best, on_trial)

    assert best is not None
    return OptimizationResult("grid_search", seed, best[0], best[1], history)


OPTIMIZERS: Dict[str, Callable[..., OptimizationResult]] = {
    "random_search": random_search,
    "local_search": local_search,
    "grid_search": grid_search,
}


def default_search_space(features: Sequence[str] = FEATURE_NAMES) -> SearchSpace:
    """Non-negative weights in [0, 4], the range used by the Phase I experiments.

    Restricting weights to be non-negative encodes the prior that every feature
    is a cost-to-go proxy rather than a reward; the bound is arbitrary but
    finite, which matters because greedy search is invariant to positive
    rescaling and an unbounded box would therefore be redundant.
    """
    return SearchSpace({f: (0.0, 4.0) for f in features})


def is_degenerate(candidate: HeuristicCandidate, tolerance: float = 1e-9) -> bool:
    """Whether all weights vanish, making the heuristic uninformative."""
    return all(math.isclose(v, 0.0, abs_tol=tolerance) for v in candidate.weights.values())
