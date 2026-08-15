#!/usr/bin/env python3
"""Generates the Phase I benchmark suite.

Instances are written to ``instances/blocksworld/`` and the manifest to
``benchmarks/blocksworld.json``. Generation is a pure function of the base
seed, so the suite can be reconstructed byte-for-byte at any time; the
generated files are nevertheless kept under version control so that reported
results refer to a fixed set of problems.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from hd.benchmark import BenchmarkSuite  # noqa: E402
from hd.domains import blocksworld  # noqa: E402
from hd.paths import BENCHMARK_DIR, INSTANCE_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, nargs="+", default=[3, 4, 5, 6, 7],
                        help="block counts to generate")
    parser.add_argument("--per-size", type=int, default=4, help="instances per block count")
    parser.add_argument("--seed", type=int, default=20240101, help="base seed")
    parser.add_argument("--suite", default="blocksworld", help="suite name")
    args = parser.parse_args()

    directory = INSTANCE_DIR / args.suite
    directory.mkdir(parents=True, exist_ok=True)

    specification = blocksworld.suite_specification(
        block_counts=args.blocks, instances_per_size=args.per_size, base_seed=args.seed
    )

    paths = []
    for name, num_blocks, seed in specification:
        task = blocksworld.generate(num_blocks=num_blocks, seed=seed)
        path = directory / f"{name}.task"
        path.write_text(task.to_task_text(name), encoding="utf-8")
        paths.append(path)
        print(f"{name:>18}  blocks={num_blocks}  seed={seed}  "
              f"props={task.num_propositions}  actions={len(task.actions)}")

    suite = BenchmarkSuite(
        name=args.suite,
        instances=paths,
        description=(
            f"Blocksworld, {args.per_size} instances for each of "
            f"{len(args.blocks)} block counts, generated from base seed {args.seed}."
        ),
        metadata={
            "domain": "blocksworld",
            "generator": "hd.domains.blocksworld",
            "base_seed": args.seed,
            "block_counts": list(args.blocks),
            "instances_per_size": args.per_size,
        },
    )
    manifest = suite.save(BENCHMARK_DIR / f"{args.suite}.json")
    print(f"\n{len(paths)} instances written to {directory}")
    print(f"manifest: {manifest}")
    print(json.dumps(suite.metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
