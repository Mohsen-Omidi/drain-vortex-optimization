"""
Analyze MVDO ablation results on the CEC2017 subset.

Input:
    results/ablation_cec2017_subset_full_summary.csv

Outputs:
    results/ablation_cec2017_subset_overview.csv
    results/ablation_cec2017_subset_ranks.csv
    results/ablation_cec2017_subset_category_overview.csv
    results/ablation_cec2017_subset_wilcoxon.csv
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


INFILE = "results/ablation_cec2017_subset_full_summary.csv"
TARGET = "MVDO_full"
ALPHA = 0.05


def category(fn_id: int) -> str:
    if fn_id in [1, 3]:
        return "unimodal_basic"
    if fn_id in [4, 5, 6, 7, 8, 9, 10]:
        return "multimodal"
    if 11 <= fn_id <= 20:
        return "hybrid"
    if 21 <= fn_id <= 30:
        return "composition"
    return "other"


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
            r["category"] = category(r["fn_id"])
            rows.append(r)
    return rows


def summarize_subset(rows, label, predicate):
    sub = [r for r in rows if predicate(r)]
    variants = sorted(set(r["variant"] for r in sub))
    cases = sorted(set((r["dim"], r["fn_id"]) for r in sub))

    print("\n" + "=" * 72)
    print(f"Subset: {label}")
    print(f"Cases : {len(cases)}")
    print("=" * 72)

    values = {v: [] for v in variants}
    logs = {v: [] for v in variants}
    rank_rows = []

    wins = defaultdict(int)
    rank_sums = defaultdict(float)
    log_sums = defaultdict(float)

    for dim, fn_id in cases:
        case_rows = {
            r["variant"]: r
            for r in sub
            if r["dim"] == dim and r["fn_id"] == fn_id
        }
        missing = [v for v in variants if v not in case_rows]
        if missing:
            raise RuntimeError(f"Missing variants {missing} for F{fn_id} D{dim}")

        errors = np.asarray([case_rows[v]["mean_error"] for v in variants], dtype=float)
        ranks = rankdata(errors, method="average")
        best_variant = variants[int(np.argmin(errors))]
        wins[best_variant] += 1

        for v, rank in zip(variants, ranks):
            err = case_rows[v]["mean_error"]
            logv = case_rows[v]["logscore"]
            values[v].append(err)
            logs[v].append(logv)
            rank_sums[v] += float(rank)
            log_sums[v] += float(logv)
            rank_rows.append({
                "subset": label,
                "dim": dim,
                "fn_id": fn_id,
                "category": category(fn_id),
                "variant": v,
                "mean_error": err,
                "logscore": logv,
                "rank": float(rank),
            })

    overview = []
    for v in variants:
        overview.append({
            "subset": label,
            "variant": v,
            "wins": wins[v],
            "avg_rank": rank_sums[v] / len(cases),
            "avg_log": log_sums[v] / len(cases),
        })

    overview = sorted(overview, key=lambda x: (x["avg_rank"], x["avg_log"]))

    print("\nVariant overview:")
    for r in overview:
        print(
            f"{r['variant']:<20s} | "
            f"wins={r['wins']:2d} | "
            f"avg_rank={r['avg_rank']:.3f} | "
            f"avg_log={r['avg_log']:.4f}"
        )

    if len(variants) > 2 and len(cases) > 1:
        arrays = [np.asarray(values[v], dtype=float) for v in variants]
        stat, p = friedmanchisquare(*arrays)
        print("\nFriedman:")
        print(f"statistic = {stat:.6f}")
        print(f"p-value   = {p:.6e}")

    wilcoxon_rows = []
    if TARGET in variants and len(cases) > 1:
        target = np.asarray(logs[TARGET], dtype=float)
        pairs = []

        for v in variants:
            if v == TARGET:
                continue
            other = np.asarray(logs[v], dtype=float)
            try:
                res = wilcoxon(target, other, alternative="less", zero_method="wilcox")
                p_raw = float(res.pvalue)
            except ValueError:
                p_raw = 1.0
            diff = float(np.mean(target - other))
            pairs.append((v, p_raw))
            wilcoxon_rows.append({
                "subset": label,
                "comparison": f"{TARGET} vs {v}",
                "variant": v,
                "p_raw": p_raw,
                "mean_log_diff": diff,
            })

        holm = holm_adjust(pairs)

        print(f"\nWilcoxon: {TARGET} vs ablations, alternative='less'")
        for row in sorted(wilcoxon_rows, key=lambda x: x["p_raw"]):
            _, p_holm, sig = holm[row["variant"]]
            row["p_holm"] = p_holm
            row["sig_holm"] = sig
            print(
                f"{TARGET} vs {row['variant']:<18s} | "
                f"p_raw={row['p_raw']:.3e} | "
                f"p_holm={p_holm:.3e} | "
                f"sig={sig} | "
                f"mean_log_diff={row['mean_log_diff']:.3f}"
            )

    return overview, rank_rows, wilcoxon_rows


def main():
    rows = read_rows()

    all_overview = []
    all_ranks = []
    all_wilcox = []

    subsets = [
        ("overall", lambda r: True),
        ("D30", lambda r: r["dim"] == 30),
        ("D50", lambda r: r["dim"] == 50),
        ("unimodal_basic", lambda r: r["category"] == "unimodal_basic"),
        ("multimodal", lambda r: r["category"] == "multimodal"),
        ("hybrid", lambda r: r["category"] == "hybrid"),
        ("composition", lambda r: r["category"] == "composition"),
    ]

    for label, pred in subsets:
        ov, rr, ww = summarize_subset(rows, label, pred)
        all_overview.extend(ov)
        all_ranks.extend(rr)
        all_wilcox.extend(ww)

    with open("results/ablation_cec2017_subset_overview.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_overview[0].keys()))
        writer.writeheader()
        writer.writerows(all_overview)

    with open("results/ablation_cec2017_subset_ranks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_ranks[0].keys()))
        writer.writeheader()
        writer.writerows(all_ranks)

    if all_wilcox:
        fieldnames = ["subset", "comparison", "variant", "p_raw", "p_holm", "sig_holm", "mean_log_diff"]
        with open("results/ablation_cec2017_subset_wilcoxon.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_wilcox)

    print("\nSaved:")
    print("  results/ablation_cec2017_subset_overview.csv")
    print("  results/ablation_cec2017_subset_ranks.csv")
    print("  results/ablation_cec2017_subset_wilcoxon.csv")


if __name__ == "__main__":
    main()
