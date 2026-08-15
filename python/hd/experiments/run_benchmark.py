#!/usr/bin/env python3
"""Baseline benchmark run.

Evaluates the built-in heuristics on a benchmark suite under one search
algorithm and writes an experiment record. This establishes the reference
points against which discovered heuristics are reported.

    python -m hd.experiments.run_benchmark --search gbfs
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from ..benchmark import BenchmarkSuite, comparison_table
from ..candidate import BASELINE_HEURISTICS
from ..experiment import Evaluation, Experiment, build_record
from ..objective import ObjectiveWeights
from ..planner import Planner


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="blocksworld", help="suite name or manifest path")
    parser.add_argument("--search", default="gbfs", choices=("bfs", "gbfs", "astar"))
    parser.add_argument("--heuristics", nargs="+", default=list(BASELINE_HEURISTICS))
    parser.add_argument("--max-expansions", type=int, default=200_000)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-instance", action="store_true", help="print a per-instance table")
    parser.add_argument("--no-save", action="store_true", help="do not write a result record")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    suite = BenchmarkSuite.load(args.suite)
    experiment = Experiment(
        suite=suite,
        planner=Planner(),
        weights=ObjectiveWeights(),
        search=args.search,
        max_expansions=args.max_expansions,
        time_limit=args.time_limit,
        seed=args.seed,
    )

    print(f"suite      {suite.name}  ({len(suite)} instances)")
    print(f"search     {args.search}")
    print(f"expansion budget {args.max_expansions}, time limit {args.time_limit}s\n")

    header = f"{'heuristic':<18}{'solved':>8}{'expanded':>12}{'generated':>12}{'cost':>10}{'time':>10}"
    print(header)
    print("-" * len(header))

    evaluations: Dict[str, Evaluation] = {}
    for name in args.heuristics:
        evaluation = experiment.evaluate(name)
        evaluations[name] = evaluation
        s = evaluation.summary
        print(
            f"{name:<18}{s.num_solved:>4}/{s.num_instances:<3}{s.total_expanded:>12}"
            f"{s.total_generated:>12}{s.total_cost:>10.0f}{s.total_runtime:>9.2f}s"
        )

    if args.per_instance:
        print()
        print(comparison_table({name: e.run for name, e in evaluations.items()}))

    if not args.no_save:
        record = build_record("benchmark", experiment, evaluations)
        print(f"\nrecord written to {record.save()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
