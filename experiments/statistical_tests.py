import csv
import math
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata


INFILE = "results/comparison_full_summary_with_hho.csv"

OUT_FRIEDMAN = "results/stat_friedman.csv"
OUT_WILCOXON = "results/stat_wilcoxon_mvdo_vs_baselines.csv"

ALPHA = 0.05
TARGET = "MVDO"


def holm_adjust(pairs):
    """
    pairs: list of (name, pvalue)
    returns dict name -> (raw_p, holm_p, significant)
    """
    m = len(pairs)
    sorted_pairs = sorted(pairs, key=lambda x: x[1])

    adjusted = {}
    running_max = 0.0

    for i, (name, p) in enumerate(sorted_pairs):
        factor = m - i
        adj = min(1.0, factor * p)
        running_max = max(running_max, adj)
        adjusted[name] = {
            "raw_p": p,
            "holm_p": running_max,
            "significant": running_max < ALPHA,
        }

    return adjusted


def main():
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

    algorithms = sorted(set(r["algorithm"] for r in rows))
    cases = sorted(set((r["dim"], r["fn_id"]) for r in rows))

    print("Algorithms:", algorithms)
    print("Cases:", len(cases))

    # Build matrix: rows = cases, cols = algorithms
    data = {alg: [] for alg in algorithms}
    logdata = {alg: [] for alg in algorithms}

    for case in cases:
        dim, fn_id = case
        case_rows = {r["algorithm"]: r for r in rows if r["dim"] == dim and r["fn_id"] == fn_id}

        missing = [alg for alg in algorithms if alg not in case_rows]
        if missing:
            raise RuntimeError(f"Missing algorithms {missing} for case {case}")

        for alg in algorithms:
            err = case_rows[alg]["mean_error"]
            data[alg].append(err)
            logdata[alg].append(math.log10(err + 1e-12))

    # Friedman test on mean errors
    arrays = [np.asarray(data[alg], dtype=float) for alg in algorithms]
    stat, p = friedmanchisquare(*arrays)

    # Average ranks per case, lower error = better rank
    rank_sums = defaultdict(float)
    wins = defaultdict(int)

    for idx, case in enumerate(cases):
        errors = np.asarray([data[alg][idx] for alg in algorithms], dtype=float)
        ranks = rankdata(errors, method="average")  # lower error gets rank 1

        best_idx = int(np.argmin(errors))
        wins[algorithms[best_idx]] += 1

        for alg, rank in zip(algorithms, ranks):
            rank_sums[alg] += float(rank)

    avg_ranks = {alg: rank_sums[alg] / len(cases) for alg in algorithms}

    print("\n=== Friedman test ===")
    print(f"Statistic = {stat:.6f}")
    print(f"p-value   = {p:.6e}")

    print("\n=== Average ranks ===")
    for alg in sorted(algorithms, key=lambda a: avg_ranks[a]):
        print(f"{alg:>6s} | avg_rank={avg_ranks[alg]:.3f} | wins={wins[alg]:2d}")

    with open(OUT_FRIEDMAN, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algorithm", "average_rank", "wins", "friedman_statistic", "friedman_p_value", "num_cases"]
        )
        writer.writeheader()
        for alg in sorted(algorithms, key=lambda a: avg_ranks[a]):
            writer.writerow({
                "algorithm": alg,
                "average_rank": avg_ranks[alg],
                "wins": wins[alg],
                "friedman_statistic": stat,
                "friedman_p_value": p,
                "num_cases": len(cases),
            })

    # Wilcoxon signed-rank tests: MVDO vs each baseline over the 24 cases.
    # Use log10(mean_error + eps) for numerical stability across very different scales.
    mvdo = np.asarray(logdata[TARGET], dtype=float)

    wilcox_results = []
    raw_p_pairs = []

    for alg in algorithms:
        if alg == TARGET:
            continue

        other = np.asarray(logdata[alg], dtype=float)

        # alternative="less" tests whether MVDO errors are statistically smaller.
        res_less = wilcoxon(mvdo, other, alternative="less", zero_method="wilcox")
        res_two = wilcoxon(mvdo, other, alternative="two-sided", zero_method="wilcox")

        diff = mvdo - other

        item = {
            "comparison": f"{TARGET}_vs_{alg}",
            "baseline": alg,
            "mean_log_error_mvdo": float(np.mean(mvdo)),
            "mean_log_error_baseline": float(np.mean(other)),
            "mean_log_difference_mvdo_minus_baseline": float(np.mean(diff)),
            "wilcoxon_stat_less": float(res_less.statistic),
            "p_less_raw": float(res_less.pvalue),
            "wilcoxon_stat_two_sided": float(res_two.statistic),
            "p_two_sided_raw": float(res_two.pvalue),
        }
        wilcox_results.append(item)
        raw_p_pairs.append((alg, float(res_less.pvalue)))

    holm = holm_adjust(raw_p_pairs)

    print("\n=== Wilcoxon: MVDO vs baselines, alternative='less' ===")
    for item in sorted(wilcox_results, key=lambda x: x["p_less_raw"]):
        alg = item["baseline"]
        item["p_less_holm"] = holm[alg]["holm_p"]
        item["significant_after_holm"] = holm[alg]["significant"]

        print(
            f"MVDO vs {alg:>4s} | "
            f"p_raw={item['p_less_raw']:.3e} | "
            f"p_holm={item['p_less_holm']:.3e} | "
            f"sig={item['significant_after_holm']} | "
            f"mean_log_diff={item['mean_log_difference_mvdo_minus_baseline']:.3f}"
        )

    with open(OUT_WILCOXON, "w", newline="") as f:
        fieldnames = [
            "comparison",
            "baseline",
            "mean_log_error_mvdo",
            "mean_log_error_baseline",
            "mean_log_difference_mvdo_minus_baseline",
            "wilcoxon_stat_less",
            "p_less_raw",
            "p_less_holm",
            "significant_after_holm",
            "wilcoxon_stat_two_sided",
            "p_two_sided_raw",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(wilcox_results, key=lambda x: x["p_less_raw"]):
            writer.writerow(item)

    print("\nSaved:")
    print(f"  {OUT_FRIEDMAN}")
    print(f"  {OUT_WILCOXON}")


if __name__ == "__main__":
    main()
