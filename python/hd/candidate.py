"""Candidate heuristics.

A candidate is the explicit linear form

    h_theta(s) = sum_i theta_i f_i(s)

represented as a mapping from feature name to weight. The representation is
deliberately transparent: it can be printed as an equation, serialised to
JSON or YAML, and handed back to the planner verbatim, so any reported result
can be reproduced from the record alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

#: Features exposed by the C++ engine. Kept in sync with ``cpp/include/hd/features.hpp``;
#: :func:`hd.planner.Planner.features` queries the binary, and the test suite
#: asserts that the two agree.
FEATURE_NAMES = (
    "unsatisfied_goals",
    "achieved_goals",
    "applicable_actions",
    "true_propositions",
    "relaxed_layers",
    "relaxed_sum",
)

#: Baseline heuristics built into the engine, addressed by name.
BASELINE_HEURISTICS = ("zero", "goal_count", "relaxed_layers")


@dataclass(frozen=True)
class HeuristicCandidate:
    """A named linear heuristic over state features."""

    weights: Dict[str, float]
    name: str = "candidate"
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.weights) - set(FEATURE_NAMES))
        if unknown:
            raise ValueError(f"unknown feature(s): {', '.join(unknown)}")
        object.__setattr__(self, "weights", {k: float(v) for k, v in self.weights.items()})

    # --- planner interface ------------------------------------------------
    def to_spec(self, tolerance: float = 1e-12) -> str:
        """Renders the ``--heuristic`` argument understood by ``hd_plan``.

        Weights at or below ``tolerance`` are dropped: an unused feature is
        then never evaluated during search.
        """
        terms = [f"{k}={v!r}" for k, v in self.weights.items() if abs(v) > tolerance]
        return "linear:" + ",".join(terms) if terms else "zero"

    # --- human-readable form ---------------------------------------------
    def to_equation(self, precision: int = 2, tolerance: float = 1e-12) -> str:
        """``h(s) = 1.82 * unsatisfied_goals(s) + 0.37 * applicable_actions(s)``"""
        terms = [(k, v) for k, v in self.weights.items() if abs(v) > tolerance]
        if not terms:
            return "h(s) = 0"
        rendered = f"h(s) = {terms[0][1]:.{precision}f} * {terms[0][0]}(s)"
        for feature, weight in terms[1:]:
            sign = "-" if weight < 0 else "+"
            rendered += f"\n       {sign} {abs(weight):.{precision}f} * {feature}(s)"
        return rendered

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.to_equation()

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "weights": dict(self.weights), "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HeuristicCandidate":
        weights = data.get("weights", {})
        if not isinstance(weights, Mapping):
            raise ValueError("'weights' must be a mapping from feature name to weight")
        metadata = data.get("metadata", {})
        return cls(
            weights={str(k): float(v) for k, v in weights.items()},
            name=str(data.get("name", "candidate")),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    def save(self, path: Path | str) -> Path:
        """Writes the candidate to JSON, or to YAML when the suffix says so."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            path.write_text(_dump_yaml(self.to_dict()), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "HeuristicCandidate":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            return cls.from_dict(_load_yaml(text))
        return cls.from_dict(json.loads(text))

    # --- algebra used by the optimisers -----------------------------------
    def with_weights(self, weights: Mapping[str, float], name: Optional[str] = None):
        return HeuristicCandidate(
            weights=dict(weights), name=name or self.name, metadata=dict(self.metadata)
        )

    def normalised(self) -> "HeuristicCandidate":
        """Scales the weights to unit maximum magnitude.

        Greedy best-first search is invariant under a positive rescaling of h,
        so normalising makes candidates comparable without changing behaviour.
        """
        largest = max((abs(v) for v in self.weights.values()), default=0.0)
        if largest <= 0.0:
            return self
        return self.with_weights({k: v / largest for k, v in self.weights.items()})


def baseline_candidate(name: str) -> str:
    """Validates and returns the name of a built-in baseline heuristic."""
    if name not in BASELINE_HEURISTICS:
        raise ValueError(f"unknown baseline heuristic '{name}'")
    return name


def zero_candidate(features: Iterable[str] = FEATURE_NAMES) -> HeuristicCandidate:
    return HeuristicCandidate({f: 0.0 for f in features}, name="zero")


# --- optional YAML support ------------------------------------------------
# PyYAML is not a dependency of this project; the flat schema of a candidate is
# small enough to emit and read without it.


def _dump_yaml(data: Mapping[str, object]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(dict(data), sort_keys=False)
    except ImportError:
        lines = [f"name: {data['name']}", "weights:"]
        for key, value in dict(data["weights"]).items():  # type: ignore[arg-type]
            lines.append(f"  {key}: {value!r}")
        metadata = dict(data.get("metadata", {}))  # type: ignore[arg-type]
        lines.append("metadata:" if metadata else "metadata: {}")
        for key, value in metadata.items():
            lines.append(f"  {key}: {value!r}")
        return "\n".join(lines) + "\n"


def _load_yaml(text: str) -> Dict[str, object]:
    try:
        import yaml  # type: ignore

        return dict(yaml.safe_load(text))
    except ImportError:
        pass

    data: Dict[str, object] = {"name": "candidate", "weights": {}, "metadata": {}}
    section = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            key, _, value = raw.partition(":")
            key, value = key.strip(), value.strip()
            if value and value != "{}":
                data[key] = value
            section = key
        elif section in ("weights", "metadata"):
            key, _, value = raw.partition(":")
            target = data[section]
            assert isinstance(target, dict)
            target[key.strip()] = float(value) if section == "weights" else value.strip()
    return data
