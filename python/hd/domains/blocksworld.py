"""Propositional encoding of Blocksworld.

Blocksworld is used as the Phase I benchmark family because it is small enough
to encode exactly in propositional STRIPS, hard enough that heuristic quality
dominates search effort, and standard enough that results are comparable with
the planning literature.

Propositions (n blocks):
    on(i, j)     block i rests on block j, i != j     n(n-1)
    ontable(i)   block i rests on the table           n
    clear(i)     nothing rests on block i             n
    holding(i)   the gripper holds block i            n
    handempty    the gripper holds nothing            1

Actions: pickup, putdown, stack, unstack, each of unit cost.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# A configuration is a tower list, e.g. [[0, 1], [2]] means block 1 on block 0,
# both on the table, and block 2 alone on the table.
Configuration = List[List[int]]


@dataclass(frozen=True)
class BlocksworldEncoding:
    """Propositional encoding of one Blocksworld task."""

    num_blocks: int
    proposition_names: List[str]
    initial: List[int]
    goal: List[int]
    actions: List[Tuple[str, List[int], List[int], List[int], float]]

    @property
    def num_propositions(self) -> int:
        return len(self.proposition_names)

    def to_task_text(self, name: str) -> str:
        """Renders the task in the line-oriented format read by ``hd_plan``."""
        lines = [f"name {name}", f"propositions {self.num_propositions}"]
        for index, prop in enumerate(self.proposition_names):
            lines.append(f"prop {index} {prop}")
        lines.append("init " + " ".join(str(p) for p in sorted(self.initial)))
        lines.append("goal " + " ".join(str(p) for p in sorted(self.goal)))
        for action_name, pre, add, delete, cost in self.actions:
            lines.append(f"action {action_name} {cost:g}")
            lines.append("pre " + " ".join(str(p) for p in sorted(pre)))
            lines.append("add " + " ".join(str(p) for p in sorted(add)))
            if delete:
                lines.append("del " + " ".join(str(p) for p in sorted(delete)))
            lines.append("end")
        return "\n".join(lines) + "\n"


class _PropositionIndex:
    """Assigns a stable integer to every ground proposition."""

    def __init__(self, num_blocks: int) -> None:
        self.names: List[str] = []
        self._index: Dict[str, int] = {}
        for i in range(num_blocks):
            for j in range(num_blocks):
                if i != j:
                    self._add(f"on_{i}_{j}")
        for i in range(num_blocks):
            self._add(f"ontable_{i}")
        for i in range(num_blocks):
            self._add(f"clear_{i}")
        for i in range(num_blocks):
            self._add(f"holding_{i}")
        self._add("handempty")

    def _add(self, name: str) -> int:
        self._index[name] = len(self.names)
        self.names.append(name)
        return self._index[name]

    def __call__(self, name: str) -> int:
        return self._index[name]


def random_configuration(num_blocks: int, rng: random.Random) -> Configuration:
    """Samples a random tower decomposition of ``num_blocks`` blocks."""
    blocks = list(range(num_blocks))
    rng.shuffle(blocks)
    towers: Configuration = []
    for block in blocks:
        # Either start a new tower or extend an existing one, biased towards
        # stacking so that instances are not trivially flat.
        if towers and rng.random() < 0.6:
            towers[rng.randrange(len(towers))].append(block)
        else:
            towers.append([block])
    return towers


def configuration_propositions(config: Configuration, index: _PropositionIndex) -> List[int]:
    """Propositions true in a hand-empty state with the given towers."""
    props = [index("handempty")]
    for tower in config:
        props.append(index(f"ontable_{tower[0]}"))
        for below, above in zip(tower, tower[1:]):
            props.append(index(f"on_{above}_{below}"))
        props.append(index(f"clear_{tower[-1]}"))
    return props


def goal_propositions(config: Configuration, index: _PropositionIndex) -> List[int]:
    """Goal conditions describing the target towers.

    Only the ``on`` and ``ontable`` atoms are required: ``clear`` and
    ``handempty`` follow from them, and leaving them out keeps the goal a
    genuine partial specification.
    """
    props: List[int] = []
    for tower in config:
        props.append(index(f"ontable_{tower[0]}"))
        for below, above in zip(tower, tower[1:]):
            props.append(index(f"on_{above}_{below}"))
    return props


def build_actions(
    num_blocks: int, index: _PropositionIndex
) -> List[Tuple[str, List[int], List[int], List[int], float]]:
    actions = []
    for i in range(num_blocks):
        actions.append(
            (
                f"pickup_{i}",
                [index(f"ontable_{i}"), index(f"clear_{i}"), index("handempty")],
                [index(f"holding_{i}")],
                [index(f"ontable_{i}"), index(f"clear_{i}"), index("handempty")],
                1.0,
            )
        )
        actions.append(
            (
                f"putdown_{i}",
                [index(f"holding_{i}")],
                [index(f"ontable_{i}"), index(f"clear_{i}"), index("handempty")],
                [index(f"holding_{i}")],
                1.0,
            )
        )
        for j in range(num_blocks):
            if i == j:
                continue
            actions.append(
                (
                    f"stack_{i}_{j}",
                    [index(f"holding_{i}"), index(f"clear_{j}")],
                    [index(f"on_{i}_{j}"), index(f"clear_{i}"), index("handempty")],
                    [index(f"holding_{i}"), index(f"clear_{j}")],
                    1.0,
                )
            )
            actions.append(
                (
                    f"unstack_{i}_{j}",
                    [index(f"on_{i}_{j}"), index(f"clear_{i}"), index("handempty")],
                    [index(f"holding_{i}"), index(f"clear_{j}")],
                    [index(f"on_{i}_{j}"), index(f"clear_{i}"), index("handempty")],
                    1.0,
                )
            )
    return actions


def generate(num_blocks: int, seed: int) -> BlocksworldEncoding:
    """Generates one solvable Blocksworld task with a reproducible seed."""
    if num_blocks < 2:
        raise ValueError("Blocksworld needs at least two blocks")
    rng = random.Random(seed)
    index = _PropositionIndex(num_blocks)

    initial_config = random_configuration(num_blocks, rng)
    goal_config = random_configuration(num_blocks, rng)
    # Reject a goal that already holds in the initial state.
    for _ in range(32):
        if sorted(map(sorted, goal_config)) != sorted(map(sorted, initial_config)):
            break
        goal_config = random_configuration(num_blocks, rng)

    return BlocksworldEncoding(
        num_blocks=num_blocks,
        proposition_names=index.names,
        initial=configuration_propositions(initial_config, index),
        goal=goal_propositions(goal_config, index),
        actions=build_actions(num_blocks, index),
    )


def suite_specification(
    block_counts: Sequence[int] = (3, 4, 5, 6, 7),
    instances_per_size: int = 4,
    base_seed: int = 20240101,
) -> List[Tuple[str, int, int]]:
    """Enumerates ``(name, num_blocks, seed)`` triples for a benchmark suite."""
    spec = []
    for num_blocks in block_counts:
        for k in range(instances_per_size):
            seed = base_seed + 1000 * num_blocks + k
            spec.append((f"blocks-{num_blocks:02d}-{k:02d}", num_blocks, seed))
    return spec
