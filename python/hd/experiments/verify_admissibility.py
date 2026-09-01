#!/usr/bin/env python3
"""Admissibility verification of heuristics against exact goal distances.

Enumerates the reachable state space of every instance small enough to allow
it, computes h* exactly, and reports whether each heuristic ever overestimates.
Instances beyond the enumeration ceiling are left unchecked and reported as
such; a clean verdict is therefore evidence and not proof, while a violation
comes with the witness state that establishes it.

    python -m hd.experiments.verify_admissibility --max-states 200000
    python -m hd.experiments.verify_admissibility --candidate results/best.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ..admissibility import (
    DEFAULT_MAX_STATES,
    Verifier,
    VerificationRun,
    summary_table,
)
from ..benchmark import BenchmarkSuite
from ..candidate import BASELINE_HEURISTICS, HeuristicCandidate
from ..experiment import utc_timestamp
from ..paths import RESULTS_DIR
from ..planner import HeuristicLike


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="blocksworld", help="suite name or manifest path")
    parser.add_argument(
        "--heuristics",
        nargs="+",
        default=list(BASELINE_HEURISTICS),
        help="built-in baselines or linear specifications to check",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="FILE",
        help="candidate file to check alongside the baselines (repeatable)",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=DEFAULT_MAX_STATES,
        help="enumeration ceiling per instance; larger instances are skipped",
    )
    parser.add_argument(
        "--time-limit", type=float, default=0.0, help="enumeration budget per instance, 0 = none"
    )
    parser.add_argument("--witnesses", type=int, default=3, help="witnesses kept per heuristic")
    parser.add_argument("--limit", type=int, default=0, help="check only the first N instances")
    parser.add_argument("--per-instance", action="store_true", help="print a per-instance table")
    parser.add_argument("--no-save", action="store_true", help="do not write a result record")
    return parser.parse_args(argv)


def per_instance_table(run: VerificationRun) -> str:
    header = f"{'instance':<18}{'states':>10}{'max h*':>9}  " + "  ".join(
        f"{h:>22}" for h in run.heuristics
    )
    lines = [header, "-" * len(header)]
    for instance in run.instances:
        space = instance.state_space
        if not instance.checked:
            lines.append(f"{instance.instance:<18}{space.states:>10}{'-':>9}  {space.status}")
            continue
        cells = []
        for name in run.heuristics:
            report = instance.reports[name]
            cells.append(
                f"{report.admissibility_violations:>6} viol {report.mean_informedness:>7.3f}"
                if not report.admissible
                else f"{'ok':>6}      {report.mean_informedness:>7.3f}"
            )
        lines.append(
            f"{instance.instance:<18}{space.states:>10}{space.max_h_star:>9.0f}  "
            + "  ".join(f"{c:>22}" for c in cells)
        )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    suite = BenchmarkSuite.load(args.suite)
    instances = suite.instances[: args.limit] if args.limit else suite.instances

    heuristics: List[HeuristicLike] = list(args.heuristics)
    heuristics += [HeuristicCandidate.load(path) for path in args.candidate]

    print(f"suite      {suite.name}  ({len(instances)} instances)")
    print(f"ceiling    {args.max_states} states per instance\n")

    run = Verifier().verify(
        instances,
        heuristics,
        max_states=args.max_states,
        time_limit=args.time_limit,
        witnesses=args.witnesses,
    )
    print(summary_table(run))

    if args.per_instance:
        print()
        print(per_instance_table(run))

    violations = [s for s in run.summaries() if s.witness is not None]
    if violations:
        print("\nworst witness per inadmissible heuristic:")
        for summary in violations:
            assert summary.witness is not None
            print(f"  {summary.heuristic:<22}{summary.witness.describe()}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = utc_timestamp()
        document = {"schema": "hd.verification/1", "suite": suite.name, **run.to_dict()}
        path = RESULTS_DIR / f"verification-{timestamp.replace(':', '').replace('-', '')}.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nrecord written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
