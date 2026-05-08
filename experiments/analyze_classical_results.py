import csv
import math
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata


INFILE = "results/classical_full_summary.csv"
TARGET = "MVDO"
ALPHA = 0.05


def read_rows():
    rows = []
    with open(INFILE, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["fn_id"] = int(r["fn_id"])
            r["dim"] = int(r["dim"])
            r["mean_error"] = float(r["mean_error"])
            r["std_error"] = float(r["std_error"])
            r["median_error"] = float(r["median_error"])
            r["logscore"] = float(r["logscore"])
            rows.append(r)
    return rows


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


def analyze_subset(rows, label, predicate):
    sub = [r for r in rows if predicate(r)]
    algorithms = sorted(set(r["algorithm"] for r in sub))
    cases = sorted(set((r["dim"], r["fn_id"]) for r in sub))

    print("\n" + "=" * 70)
    print(f"Subset: {label}")
    print(f"Cases : {len(cases)}")
    print("=" * 70)

    data = {alg: [] for alg in algorithms}
    logdata = {alg: [] for alg in algorithms}

    for dim, fn_id in cases:
        case_rows = {
            r["algorithm"]: r
            for r in sub
            if r["dim"] == dim and r["fn_id"] == fn_id
        }

        missing = [alg for alg in algorithms if alg not in case_rows]
        if missing:
            raise RuntimeError(f"Missing {missing} for case D={dim}, F{fn_id}")

        for alg in algorithms:
            err = case_rows[alg]["mean_error"]
            data[alg].append(err)
            logdata[alg].append(math.log10(err + 1e-12))

    rank_sums = defaultdict(float)
    wins = defaultdict(int)
    log_sums = defaultdict(float)

    for i, case in enumerate(cases):
        errors = np.asarray([data[alg][i] for alg in algorithms])
        ranks = rankdata(errors, method="average")
        best_alg = algorithms[int(np.argmin(errors))]
        wins[best_alg] += 1

        for alg, rank in zip(algorithms, ranks):
            rank_sums[alg] += float(rank)
            log_sums[alg] += float(logdata[alg][i])

    overview = []
    for alg in algorithms:
        overview.append({
            "algorithm": alg,
            "wins": wins[alg],
            "avg_rank": rank_sums[alg] / len(cases),
            "avg_log": log_sums[alg] / len(cases),
        })

    overview = sorted(overview, key=lambda x: (x["avg_rank"], x["avg_log"]))

    print("\nAlgorithm overview:")
    for r in overview:
        print(
            f"{r['algorithm']:>6s} | "
            f"wins={r['wins']:2d} | "
            f"avg_rank={r['avg_rank']:.3f} | "
            f"avg_log={r['avg_log']:.4f}"
        )

    if len(algorithms) > 2 and len(cases) > 1:
        arrays = [np.asarray(data[alg], dtype=float) for alg in algorithms]
        stat, p = friedmanchisquare(*arrays)
        print("\nFriedman:")
        print(f"statistic = {stat:.6f}")
        print(f"p-value   = {p:.6e}")

    if TARGET in algorithms:
        mvdo = np.asarray(logdata[TARGET], dtype=float)
        pairs = []
        wilcox_items = []

        for alg in algorithms:
            if alg == TARGET:
                continue

            other = np.asarray(logdata[alg], dtype=float)
            res = wilcoxon(mvdo, other, alternative="less", zero_method="wilcox")
            diff = float(np.mean(mvdo - other))
            pairs.append((alg, float(res.pvalue)))
            wilcox_items.append((alg, float(res.pvalue), diff))

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

    out_prefix = label.lower().replace(" ", "_")
    with open(f"results/classical_overview_{out_prefix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "wins", "avg_rank", "avg_log"])
        writer.writeheader()
        writer.writerows(overview)


def main():
    rows = read_rows()

    analyze_subset(
        rows,
        "overall",
        lambda r: True,
    )

    analyze_subset(
        rows,
        "scalable_all",
        lambda r: r["fn_id"] <= 13,
    )

    analyze_subset(
        rows,
        "scalable_D30",
        lambda r: r["fn_id"] <= 13 and r["dim"] == 30,
    )

    analyze_subset(
        rows,
        "scalable_D100",
        lambda r: r["fn_id"] <= 13 and r["dim"] == 100,
    )

    analyze_subset(
        rows,
        "fixed_dimensional",
        lambda r: r["fn_id"] >= 14,
    )


if __name__ == "__main__":
    main()
