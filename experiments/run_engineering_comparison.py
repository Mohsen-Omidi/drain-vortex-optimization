"""
Run MVDO and baselines on constrained engineering design problems.

Smoke:
    python experiments/run_engineering_comparison.py --mode smoke

Full:
    python experiments/run_engineering_comparison.py --mode full --algorithms MVDO PSO GWO WOA SCA AOA EO

The objective minimized by the optimizers is a quadratic-penalty score.
The script reports raw objective, penalized score, violation, and feasibility.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, Type

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.mvdo import MVDO
from algorithms.baselines import PSO, GWO, WOA, SCA, AOA, EO
from algorithms.svoa import SVOA
from benchmarks import engineering


ALGORITHMS: Dict[str, Type] = {
    "MVDO": MVDO,
    "SVOA": SVOA,
    "PSO": PSO,
    "GWO": GWO,
    "WOA": WOA,
    "SCA": SCA,
    "AOA": AOA,
    "EO": EO,
}


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
    }


def run_one(
    alg_name: str,
    problem_name: str,
    pop_size: int,
    max_iter: int,
    num_runs: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    penalty_factor: float,
    feasibility_tol: float,
):
    problem = engineering.build_problem(
        problem_name,
        device=device,
        dtype=dtype,
        penalty_factor=penalty_factor,
        feasibility_tol=feasibility_tol,
        num_runs=num_runs,
    )

    Alg = ALGORITHMS[alg_name]

    opt = Alg(
        fn=problem,
        dim=problem.dim,
        lb=problem.lb,
        ub=problem.ub,
        pop_size=pop_size,
        max_iter=max_iter,
        num_runs=num_runs,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    start = time.time()
    result = opt.run(verbose=False)
    elapsed = time.time() - start

    track = problem.tracking_as_numpy()

    if not track:
        # Fallback if tracking failed.
        score = np.asarray(result["best_fitness_per_run"], dtype=np.float64)
        raw = score.copy()
        vsum = np.full(num_runs, np.nan)
        vmax = np.full(num_runs, np.nan)
        best_x = np.full((num_runs, problem.dim), np.nan)
    else:
        score = np.asarray(track["best_score"], dtype=np.float64)
        raw = np.asarray(track["best_objective"], dtype=np.float64)
        vsum = np.asarray(track["best_violation_sum"], dtype=np.float64)
        vmax = np.asarray(track["best_violation_max"], dtype=np.float64)
        best_x = np.asarray(track["best_x"], dtype=np.float64)

    feasible = np.asarray(vmax <= feasibility_tol, dtype=bool)
    feasible_rate = float(np.mean(feasible))

    score_s = summarize(score)
    raw_s = summarize(raw)
    vmax_s = summarize(vmax)

    if np.any(feasible):
        feasible_raw = raw[feasible]
        feasible_s = summarize(feasible_raw)
        best_feasible_obj = feasible_s["min"]
        mean_feasible_obj = feasible_s["mean"]
        std_feasible_obj = feasible_s["std"]
        median_feasible_obj = feasible_s["median"]
    else:
        best_feasible_obj = float("inf")
        mean_feasible_obj = float("inf")
        std_feasible_obj = float("inf")
        median_feasible_obj = float("inf")

    # The ranking metric should penalize infeasible runs.
    # If all are feasible, this is simply the raw objective mean.
    ranking_score = mean_feasible_obj if feasible_rate == 1.0 else score_s["mean"]

    row = {
        "algorithm": alg_name,
        "suite": "Engineering",
        "problem": problem_name,
        "dim": problem.dim,
        "num_runs": num_runs,
        "pop_size": pop_size,
        "max_iter": max_iter,
        "fes_per_run_nominal": pop_size * max_iter,
        "penalty_factor": penalty_factor,
        "feasibility_tol": feasibility_tol,
        "feasible_rate": feasible_rate,
        "ranking_score": ranking_score,
        "best_feasible_obj": best_feasible_obj,
        "mean_feasible_obj": mean_feasible_obj,
        "std_feasible_obj": std_feasible_obj,
        "median_feasible_obj": median_feasible_obj,
        "mean_raw_obj": raw_s["mean"],
        "std_raw_obj": raw_s["std"],
        "best_raw_obj": raw_s["min"],
        "median_raw_obj": raw_s["median"],
        "mean_penalized": score_s["mean"],
        "std_penalized": score_s["std"],
        "best_penalized": score_s["min"],
        "median_penalized": score_s["median"],
        "mean_max_violation": vmax_s["mean"],
        "best_max_violation": vmax_s["min"],
        "median_max_violation": vmax_s["median"],
        "elapsed_sec": float(elapsed),
    }

    raw_rows = []
    for r in range(num_runs):
        raw_row = {
            "algorithm": alg_name,
            "suite": "Engineering",
            "problem": problem_name,
            "run_idx": r,
            "score": float(score[r]),
            "raw_objective": float(raw[r]),
            "violation_sum": float(vsum[r]),
            "violation_max": float(vmax[r]),
            "feasible": bool(feasible[r]),
        }
        for j in range(problem.dim):
            raw_row[f"x{j+1}"] = float(best_x[r, j])
        raw_rows.append(raw_row)

    return row, raw_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--algorithms", nargs="+", default=["MVDO", "PSO", "GWO", "WOA", "SCA", "AOA", "EO"])
    parser.add_argument("--problems", nargs="+", default=None)
    parser.add_argument("--include-gear-train", action="store_true")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--penalty-factor", type=float, default=1e10)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.float64

    if args.mode == "smoke":
        num_runs = args.num_runs if args.num_runs is not None else 5
        max_iter = args.max_iter if args.max_iter is not None else 100
        tag = "smoke"
    else:
        num_runs = args.num_runs if args.num_runs is not None else 30
        max_iter = args.max_iter if args.max_iter is not None else 1000
        tag = "full"

    problems = args.problems if args.problems is not None else engineering.list_problems(include_optional=args.include_gear_train)

    for alg in args.algorithms:
        if alg not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {alg}. Valid: {sorted(ALGORITHMS)}")

    print(f"=== Engineering design comparison: {tag} ===")
    print(f"Device    : {args.device}")
    print(f"DType     : {dtype}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Problems  : {problems}")
    print(f"Runs      : {num_runs}")
    print(f"Pop size  : {args.pop_size}")
    print(f"Max iter  : {max_iter}")
    print(f"FEs/run   : {args.pop_size * max_iter}")
    print(f"Penalty   : {args.penalty_factor:.3e}")
    print(f"Feas tol  : {args.feasibility_tol:.3e}")
    print()

    os.makedirs("results", exist_ok=True)
    summary_path = f"results/engineering_{tag}_summary.csv"
    raw_path = f"results/engineering_{tag}_raw.csv"

    summary_rows = []
    raw_rows = []
    total_start = time.time()

    for alg_name in args.algorithms:
        print("\n==============================")
        print(f"Algorithm: {alg_name}")
        print("==============================")

        for p_idx, problem_name in enumerate(problems):
            seed = args.seed + 100000 * len(alg_name) + 1000 * p_idx

            row, rr = run_one(
                alg_name=alg_name,
                problem_name=problem_name,
                pop_size=args.pop_size,
                max_iter=max_iter,
                num_runs=num_runs,
                seed=seed,
                device=args.device,
                dtype=dtype,
                penalty_factor=args.penalty_factor,
                feasibility_tol=args.feasibility_tol,
            )

            summary_rows.append(row)
            raw_rows.extend(rr)

            print(
                f"{alg_name:>4s} {problem_name:<18s} | "
                f"best_feas={row['best_feasible_obj']:.6e} | "
                f"mean_feas={row['mean_feasible_obj']:.6e} | "
                f"feas_rate={row['feasible_rate']:.2f} | "
                f"mean_vmax={row['mean_max_violation']:.3e} | "
                f"time={row['elapsed_sec']:.2f}s"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if raw_rows:
        # Union of keys because dimensions differ.
        fieldnames = []
        for rr in raw_rows:
            for k in rr.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_rows)

    print("\n=== Done ===")
    print(f"Saved summary: {summary_path}")
    print(f"Saved raw    : {raw_path}")
    print(f"Total time   : {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
