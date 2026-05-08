"""
Analyze engineering design benchmark results.

Input:
    results/engineering_full_summary.csv

Outputs:
    results/engineering_algorithm_overview.csv
    results/engineering_ranks.csv
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata


INFILE = "results/engineering_full_summary.csv"
TARGET = "MVDO"
ALPHA = 0.05


def holm_adjust(pairs):
    m = len(pairs)
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    adjusted = {}
    running_max = 0.0
    for i, (name, p) in enumerate(sorted_pairs):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        adjusted[name] = (p, running_max, running_max < ALPHA)
    return adjusted


def main():
    rows = []
    with open(INFILE, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["ranking_score"] = float(r["ranking_score"])
            r["best_feasible_obj"] = float(r["best_feasible_obj"])
            r["mean_feasible_obj"] = float(r["mean_feasible_obj"])
            r["feasible_rate"] = float(r["feasible_rate"])
            rows.append(r)

    algorithms = sorted(set(r["algorithm"] for r in rows))
    problems = sorted(set(r["problem"] for r in rows))

    data = {alg: [] for alg in algorithms}
    rank_rows = []

    wins = defaultdict(int)
    rank_sums = defaultdict(float)
    score_sums = defaultdict(float)
    log_sums = defaultdict(float)
    feasible_sums = defaultdict(float)

    for problem in problems:
        case_rows = {r["algorithm"]: r for r in rows if r["problem"] == problem}
        scores = np.asarray([case_rows[alg]["ranking_score"] for alg in algorithms], dtype=float)
        ranks = rankdata(scores, method="average")
        best_alg = algorithms[int(np.argmin(scores))]
        wins[best_alg] += 1

        for alg, rank, score in zip(algorithms, ranks, scores):
            data[alg].append(score)
            rank_sums[alg] += float(rank)
            score_sums[alg] += float(score)
            log_sums[alg] += float(math.log10(score + 1e-12))
            feasible_sums[alg] += case_rows[alg]["feasible_rate"]

            rank_rows.append({
                "problem": problem,
                "algorithm": alg,
                "ranking_score": score,
                "rank": float(rank),
                "feasible_rate": case_rows[alg]["feasible_rate"],
                "best_feasible_obj": case_rows[alg]["best_feasible_obj"],
                "mean_feasible_obj": case_rows[alg]["mean_feasible_obj"],
            })

    overview = []
    for alg in algorithms:
        overview.append({
            "algorithm": alg,
            "wins": wins[alg],
            "avg_rank": rank_sums[alg] / len(problems),
            "avg_score": score_sums[alg] / len(problems),
            "avg_log_score": log_sums[alg] / len(problems),
            "avg_feasible_rate": feasible_sums[alg] / len(problems),
        })

    overview = sorted(overview, key=lambda x: (x["avg_rank"], x["avg_log_score"]))

    print("=== Engineering overview ===")
    for r in overview:
        print(
            f"{r['algorithm']:>6s} | "
            f"wins={r['wins']:2d} | "
            f"avg_rank={r['avg_rank']:.3f} | "
            f"avg_log_score={r['avg_log_score']:.4f} | "
            f"avg_feas={r['avg_feasible_rate']:.2f}"
        )

    print("\nPer-problem best feasible / mean feasible:")
    for problem in problems:
        print(f"\n{problem}:")
        for alg in algorithms:
            r = [x for x in rows if x["problem"] == problem and x["algorithm"] == alg][0]
            print(
                f"  {alg:>6s}: best={r['best_feasible_obj']:.6e}, "
                f"mean={r['mean_feasible_obj']:.6e}, feas={r['feasible_rate']:.2f}"
            )

    if len(algorithms) > 2 and len(problems) > 1:
        arrays = [np.asarray(data[alg], dtype=float) for alg in algorithms]
        stat, p = friedmanchisquare(*arrays)
        print("\nFriedman:")
        print(f"statistic = {stat:.6f}")
        print(f"p-value   = {p:.6e}")

    if TARGET in algorithms:
        target = np.log10(np.asarray(data[TARGET], dtype=float) + 1e-12)
        pairs = []
        wilcox_items = []
        for alg in algorithms:
            if alg == TARGET:
                continue
            other = np.log10(np.asarray(data[alg], dtype=float) + 1e-12)
            try:
                res = wilcoxon(target, other, alternative="less", zero_method="wilcox")
                p_raw = float(res.pvalue)
            except ValueError:
                p_raw = 1.0
            diff = float(np.mean(target - other))
            pairs.append((alg, p_raw))
            wilcox_items.append((alg, p_raw, diff))

        holm = holm_adjust(pairs)
        print("\nWilcoxon: MVDO vs baselines, alternative='less'")
        for alg, p_raw, diff in sorted(wilcox_items, key=lambda x: x[1]):
            _, p_holm, sig = holm[alg]
            print(
                f"MVDO vs {alg:>4s} | "
                f"p_raw={p_raw:.3e} | "
                f"p_holm={p_holm:.3e} | "
                f"sig={sig} | "
                f"mean_log_diff={diff:.3f}"
            )

    with open("results/engineering_algorithm_overview.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(overview[0].keys()))
        writer.writeheader()
        writer.writerows(overview)

    with open("results/engineering_ranks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rank_rows)

    print("\nSaved:")
    print("  results/engineering_algorithm_overview.csv")
    print("  results/engineering_ranks.csv")


if __name__ == "__main__":
    main()
