"""
Analyze and plot CEC2017 convergence/stability subset results.

Inputs:
    results/convergence_cec2017_subset_summary.csv
    results/convergence_cec2017_subset_raw.csv

Outputs:
    results/convergence_cec2017_subset_auc.csv
    results/convergence_cec2017_subset_final_overview.csv
    results/convergence_cec2017_subset_stability.csv
    results/figures/convergence_Fxx_Dyy.png
    results/figures/stability_Fxx_Dyy.png
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


SUMMARY = "results/convergence_cec2017_subset_summary.csv"
RAW = "results/convergence_cec2017_subset_raw.csv"
FIG_DIR = "results/figures"


def read_csv(path: str):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ffloat(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def iint(x):
    return int(float(x))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    rows = read_csv(SUMMARY)
    raw = read_csv(RAW)

    for r in rows:
        r["fn_id"] = iint(r["fn_id"])
        r["dim"] = iint(r["dim"])
        r["max_iter"] = iint(r["max_iter"])
        r["fes_per_run"] = iint(r["fes_per_run"])
        for k in [
            "mean_error", "median_error", "best_error", "std_error",
            "mean_log_error", "std_log_error", "solution_diversity_norm",
            "elapsed_sec"
        ]:
            r[k] = ffloat(r[k])

    for r in raw:
        r["fn_id"] = iint(r["fn_id"])
        r["dim"] = iint(r["dim"])
        r["max_iter"] = iint(r["max_iter"])
        r["fes_per_run"] = iint(r["fes_per_run"])
        r["run_idx"] = iint(r["run_idx"])
        r["error"] = ffloat(r["error"])
        r["log_error"] = ffloat(r["log_error"])

    algorithms = sorted(set(r["algorithm"] for r in rows))
    cases = sorted(set((r["fn_id"], r["dim"]) for r in rows))
    checkpoints = sorted(set(r["max_iter"] for r in rows))
    final_iter = max(checkpoints)

    print("=== Convergence subset analysis ===")
    print(f"Algorithms : {algorithms}")
    print(f"Cases      : {cases}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Final iter : {final_iter}")
    print()

    # AUC over mean log error, per case/algorithm.
    auc_rows = []
    final_rows = []
    stability_rows = []

    for fn_id, dim in cases:
        print("\n" + "=" * 70)
        print(f"Case: F{fn_id:02d} D{dim}")
        print("=" * 70)

        case_rows = [r for r in rows if r["fn_id"] == fn_id and r["dim"] == dim]

        # Convergence figure.
        plt.figure()
        for alg in algorithms:
            alg_rows = sorted([r for r in case_rows if r["algorithm"] == alg], key=lambda x: x["fes_per_run"])
            if not alg_rows:
                continue
            x = np.asarray([r["fes_per_run"] for r in alg_rows], dtype=float)
            y = np.asarray([r["mean_log_error"] for r in alg_rows], dtype=float)
            plt.plot(x, y, marker="o", label=alg)

            if len(x) > 1:
                auc = float(np.trapezoid(y, x) / (x[-1] - x[0]))
            else:
                auc = float(y[0])
            auc_rows.append({
                "fn_id": fn_id,
                "dim": dim,
                "algorithm": alg,
                "auc_mean_log_error": auc,
            })

        plt.xlabel("Function evaluations per run")
        plt.ylabel("Mean log10(error + 1e-12)")
        plt.title(f"CEC2017 F{fn_id:02d} D{dim} convergence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig_path = f"{FIG_DIR}/convergence_F{fn_id:02d}_D{dim}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        # Stability figure at all checkpoints: std of log error.
        plt.figure()
        for alg in algorithms:
            alg_rows = sorted([r for r in case_rows if r["algorithm"] == alg], key=lambda x: x["fes_per_run"])
            if not alg_rows:
                continue
            x = np.asarray([r["fes_per_run"] for r in alg_rows], dtype=float)
            y = np.asarray([r["std_log_error"] for r in alg_rows], dtype=float)
            plt.plot(x, y, marker="o", label=alg)
        plt.xlabel("Function evaluations per run")
        plt.ylabel("Std. of log10(error + 1e-12) across runs")
        plt.title(f"CEC2017 F{fn_id:02d} D{dim} run-to-run stability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig_path = f"{FIG_DIR}/stability_F{fn_id:02d}_D{dim}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        # Final checkpoint overview.
        final_case = [r for r in case_rows if r["max_iter"] == final_iter]
        final_case_sorted = sorted(final_case, key=lambda r: r["mean_error"])
        print("\nFinal checkpoint ranking by mean error:")
        for rank, r in enumerate(final_case_sorted, start=1):
            print(
                f"{rank:2d}. {r['algorithm']:<5s} | "
                f"mean={r['mean_error']:.3e} | "
                f"median={r['median_error']:.3e} | "
                f"best={r['best_error']:.3e} | "
                f"log={r['mean_log_error']:.3f} | "
                f"stdlog={r['std_log_error']:.3f} | "
                f"div={r['solution_diversity_norm']:.3e}"
            )
            final_rows.append({
                "fn_id": fn_id,
                "dim": dim,
                "rank": rank,
                "algorithm": r["algorithm"],
                "mean_error": r["mean_error"],
                "median_error": r["median_error"],
                "best_error": r["best_error"],
                "mean_log_error": r["mean_log_error"],
                "std_log_error": r["std_log_error"],
                "solution_diversity_norm": r["solution_diversity_norm"],
            })

        # Raw final-run stability.
        for alg in algorithms:
            alg_raw = [
                r for r in raw
                if r["fn_id"] == fn_id and r["dim"] == dim and r["algorithm"] == alg and r["max_iter"] == final_iter
            ]
            if not alg_raw:
                continue
            logs = np.asarray([r["log_error"] for r in alg_raw], dtype=float)
            stability_rows.append({
                "fn_id": fn_id,
                "dim": dim,
                "algorithm": alg,
                "final_iter": final_iter,
                "mean_log_error": float(np.mean(logs)),
                "std_log_error": float(np.std(logs)),
                "iqr_log_error": float(np.percentile(logs, 75) - np.percentile(logs, 25)),
                "min_log_error": float(np.min(logs)),
                "max_log_error": float(np.max(logs)),
            })

    # Overall AUC ranking.
    auc_by_alg = defaultdict(list)
    for r in auc_rows:
        auc_by_alg[r["algorithm"]].append(float(r["auc_mean_log_error"]))

    print("\n" + "=" * 70)
    print("Overall AUC ranking, lower is better")
    print("=" * 70)
    overview_auc = []
    for alg, vals in auc_by_alg.items():
        overview_auc.append({
            "algorithm": alg,
            "mean_auc": float(np.mean(vals)),
            "std_auc": float(np.std(vals)),
        })
    overview_auc = sorted(overview_auc, key=lambda x: x["mean_auc"])
    for rank, r in enumerate(overview_auc, start=1):
        print(f"{rank:2d}. {r['algorithm']:<5s} | mean_auc={r['mean_auc']:.4f} | std_auc={r['std_auc']:.4f}")

    with open("results/convergence_cec2017_subset_auc.csv", "w", newline="") as f:
        fieldnames = ["fn_id", "dim", "algorithm", "auc_mean_log_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(auc_rows)

    with open("results/convergence_cec2017_subset_final_overview.csv", "w", newline="") as f:
        fieldnames = [
            "fn_id", "dim", "rank", "algorithm", "mean_error", "median_error", "best_error",
            "mean_log_error", "std_log_error", "solution_diversity_norm"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    with open("results/convergence_cec2017_subset_stability.csv", "w", newline="") as f:
        fieldnames = [
            "fn_id", "dim", "algorithm", "final_iter",
            "mean_log_error", "std_log_error", "iqr_log_error", "min_log_error", "max_log_error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stability_rows)

    print("\nSaved:")
    print("  results/convergence_cec2017_subset_auc.csv")
    print("  results/convergence_cec2017_subset_final_overview.csv")
    print("  results/convergence_cec2017_subset_stability.csv")
    print(f"  {FIG_DIR}/convergence_Fxx_Dyy.png")
    print(f"  {FIG_DIR}/stability_Fxx_Dyy.png")


if __name__ == "__main__":
    main()
