#!/usr/bin/env python3
"""The Phase I heuristic discovery loop.

Searches the weight space of the linear heuristic

    h_theta(s) = sum_i theta_i f_i(s)

by repeatedly running the planner on a benchmark suite and scoring the
resulting metrics, then reports the discovered heuristic against the built-in
baselines and writes a reproducible experiment record.

    python -m hd.experiments.optimize_heuristic
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from ..benchmark import BenchmarkSuite
from ..candidate import FEATURE_NAMES, HeuristicCandidate
from ..experiment import Evaluation, Experiment, build_record
from ..objective import ObjectiveWeights
from ..optimize import OPTIMIZERS, SearchSpace, Trial, local_search
from ..planner import Planner


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="blocksworld", help="suite name or manifest path")
    parser.add_argument("--search", default="gbfs", choices=("gbfs", "astar"))
    parser.add_argument("--optimizer", default="random_search", choices=sorted(OPTIMIZERS))
    parser.add_argument("--iterations", type=int, default=24, help="candidate evaluations")
    parser.add_argument("--refine", type=int, default=0,
                        help="additional local-search steps around the best candidate")
    parser.add_argument("--seed", type=int, default=0, help="optimiser seed")
    parser.add_argument("--features", nargs="+", default=list(FEATURE_NAMES),
                        help="features spanning the search space")
    parser.add_argument("--weight-range", type=float, nargs=2, default=(0.0, 4.0),
                        metavar=("LOW", "HIGH"))
    parser.add_argument("--baseline", default="goal_count",
                        help="reference heuristic for normalisation and reporting")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight on expanded nodes")
    parser.add_argument("--beta", type=float, default=0.0, help="weight on runtime")
    parser.add_argument("--gamma", type=float, default=0.0, help="weight on solution cost")
    parser.add_argument("--delta", type=float, default=10.0, help="penalty on unsolved instances")
    parser.add_argument("--max-expansions", type=int, default=200_000)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--holdout-stride", type=int, default=0, metavar="N",
                        help="hold out every N-th instance; the optimiser never sees it, "
                             "and the discovered heuristic is reported on it separately")
    parser.add_argument("--save-candidate", default=None,
                        help="write the discovered heuristic to this JSON/YAML file")
    parser.add_argument("--no-save", action="store_true", help="do not write a result record")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-trial log")
    return parser.parse_args(argv)


def split(suite: BenchmarkSuite, stride: int) -> tuple[BenchmarkSuite, BenchmarkSuite | None]:
    """Splits the suite into a training part and a held-out part.

    The split is positional and therefore deterministic. Instances are ordered
    by size, so taking every N-th one keeps the difficulty distribution of the
    two parts comparable.
    """
    if stride < 2:
        return suite, None
    train, test = [], []
    for index, instance in enumerate(suite.instances):
        (test if index % stride == stride - 1 else train).append(instance)
    if not test or not train:
        return suite, None
    return (
        BenchmarkSuite(f"{suite.name}/train", train, suite.description, suite.metadata),
        BenchmarkSuite(f"{suite.name}/holdout", test, suite.description, suite.metadata),
    )


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    full_suite = BenchmarkSuite.load(args.suite)
    suite, holdout = split(full_suite, args.holdout_stride)
    weights = ObjectiveWeights(
        expanded=args.alpha, runtime=args.beta, cost=args.gamma, unsolved=args.delta
    )
    experiment = Experiment(
        suite=suite,
        planner=Planner(),
        weights=weights,
        search=args.search,
        max_expansions=args.max_expansions,
        time_limit=args.time_limit,
        seed=args.seed,
        reference_heuristic=args.baseline,
    )
    space = SearchSpace({f: tuple(args.weight_range) for f in args.features})

    print(f"suite       {suite.name} ({len(suite)} instances)"
          + (f" + holdout ({len(holdout)} instances)" if holdout else ""))
    print(f"search      {args.search}")
    print(f"objective   J = {args.alpha}*expanded + {args.beta}*runtime "
          f"+ {args.gamma}*cost + {args.delta}*unsolved   (normalised by '{args.baseline}')")
    print(f"optimiser   {args.optimizer}, {args.iterations} evaluations, seed {args.seed}")
    print(f"space       {len(space.features)} features in "
          f"[{args.weight_range[0]:g}, {args.weight_range[1]:g}]\n")

    def log(trial: Trial) -> None:
        if args.quiet:
            return
        mark = "*" if trial.accepted else " "
        expanded = experiment.evaluate(trial.candidate).summary.total_expanded
        print(f"  {mark} trial {trial.iteration:>3}  J = {trial.objective.value:8.4f}"
              f"  expanded = {expanded:>8}  coverage = {trial.objective.coverage:.0%}")

    result = OPTIMIZERS[args.optimizer](
        space, experiment.scorer(), seed=args.seed, on_trial=log,
        **({"iterations": args.iterations} if args.optimizer != "grid_search" else {}),
    )
    if args.refine:
        print(f"\n  refining for {args.refine} local steps")
        refined = local_search(
            space, experiment.scorer(), iterations=args.refine, seed=args.seed + 1,
            initial=result.best_candidate, on_trial=log,
        )
        result.history.extend(refined.history)
        if refined.best_objective.value < result.best_objective.value:
            result.best_candidate = refined.best_candidate
            result.best_objective = refined.best_objective
        result.optimizer += "+local_search"

    discovered = result.best_candidate.with_weights(result.best_candidate.weights, name="discovered")
    evaluations: Dict[str, Evaluation] = {
        args.baseline: experiment.evaluate(args.baseline),
        "discovered": experiment.evaluate(discovered),
    }
    baseline_summary = evaluations[args.baseline].summary
    learned_summary = evaluations["discovered"].summary

    print("\nCandidate:")
    print("    " + discovered.to_equation().replace("\n", "\n    "))
    print(f"\nBenchmark:\n    {len(suite)} instances ({suite.name})")
    print("\nResults:")
    print(f"    baseline  {args.search.upper():<6} {baseline_summary.total_expanded:>9} expanded, "
          f"{baseline_summary.num_solved}/{baseline_summary.num_instances} solved, "
          f"cost {baseline_summary.total_cost:.0f}")
    print(f"    learned   {args.search.upper():<6} {learned_summary.total_expanded:>9} expanded, "
          f"{learned_summary.num_solved}/{learned_summary.num_instances} solved, "
          f"cost {learned_summary.total_cost:.0f}")

    if baseline_summary.total_expanded > 0:
        reduction = 1.0 - learned_summary.total_expanded / baseline_summary.total_expanded
        print(f"\nImprovement:\n    {reduction:+.1%} expanded nodes "
              f"(objective J = {result.best_objective.value:.4f} after "
              f"{result.num_evaluations} evaluations)")

    if holdout is not None:
        holdout_experiment = Experiment(
            suite=holdout, planner=experiment.planner, weights=weights, search=args.search,
            max_expansions=args.max_expansions, time_limit=args.time_limit, seed=args.seed,
            reference_heuristic=args.baseline,
        )
        held_baseline = holdout_experiment.evaluate(args.baseline).summary
        held_learned = holdout_experiment.evaluate(discovered).summary
        evaluations[f"{args.baseline}@holdout"] = holdout_experiment.evaluate(args.baseline)
        evaluations["discovered@holdout"] = holdout_experiment.evaluate(discovered)
        print("\nHeld-out instances (never seen by the optimiser):")
        print(f"    baseline  {held_baseline.total_expanded:>9} expanded, "
              f"{held_baseline.num_solved}/{held_baseline.num_instances} solved")
        print(f"    learned   {held_learned.total_expanded:>9} expanded, "
              f"{held_learned.num_solved}/{held_learned.num_instances} solved")
        if held_baseline.total_expanded > 0:
            held_reduction = 1.0 - held_learned.total_expanded / held_baseline.total_expanded
            print(f"    {held_reduction:+.1%} expanded nodes")

    if args.save_candidate:
        print(f"\ncandidate written to {discovered.save(args.save_candidate)}")
    if not args.no_save:
        record = build_record("optimize_heuristic", experiment, evaluations, result, space)
        print(f"record written to {record.save()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
