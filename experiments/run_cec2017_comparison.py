"""
Run MVDO and baseline algorithms on the CEC2017 benchmark suite.

Recommended first smoke test:
    python experiments/run_cec2017_comparison.py --mode smoke

Full reference-style run:
    python experiments/run_cec2017_comparison.py --mode full --algorithms MVDO PSO GWO WOA SCA AOA EO

The script assumes:
    benchmarks/cec2017.py
    algorithms/mvdo.py
    algorithms/baselines.py

CEC2017 functions are evaluated through the cec2017-py package.
Install once in your conda environment:
    pip install git+https://github.com/tilleyd/cec2017-py.git
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
from benchmarks import cec2017


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


def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "logscore": float(np.mean(np.log10(x + 1e-12))),
    }


def run_one(
    alg_name: str,
    fn_id: int,
    dim: int,
    pop_size: int,
    max_iter: int,
    num_runs: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    fn = cec2017.build_function(fn_id=fn_id, dim=dim, device=device, dtype=dtype)
    Alg = ALGORITHMS[alg_name]

    opt = Alg(
        fn=fn,
        dim=dim,
        lb=fn.lb,
        ub=fn.ub,
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

    best = np.asarray(result["best_fitness_per_run"], dtype=np.float64)
    err = np.maximum(best - float(fn.bias), 0.0)
    s = summarize(err)

    row = {
        "algorithm": alg_name,
        "suite": "CEC2017",
        "fn_id": fn_id,
        "dim": dim,
        "num_runs": num_runs,
        "pop_size": pop_size,
        "max_iter": max_iter,
        "fes_per_run_nominal": pop_size * max_iter,
        "mean_error": s["mean"],
        "std_error": s["std"],
        "min_error": s["min"],
        "median_error": s["median"],
        "max_error": s["max"],
        "logscore": s["logscore"],
        "mean_fitness": float(np.mean(best)),
        "std_fitness": float(np.std(best)),
        "min_fitness": float(np.min(best)),
        "median_fitness": float(np.median(best)),
        "max_fitness": float(np.max(best)),
        "elapsed_sec": float(elapsed),
    }

    return row, best.tolist(), err.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--algorithms", nargs="+", default=["MVDO", "PSO", "GWO", "WOA", "SCA", "AOA", "EO"])
    parser.add_argument("--dims", nargs="+", type=int, default=None)
    parser.add_argument("--functions", nargs="+", type=int, default=None)
    parser.add_argument("--include-f2", action="store_true")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dtype = torch.float64

    if args.mode == "smoke":
        dims = args.dims if args.dims is not None else [30]
        functions = args.functions if args.functions is not None else [1, 3, 5, 10, 15, 20, 25, 30]
        num_runs = args.num_runs if args.num_runs is not None else 5
        max_iter = args.max_iter if args.max_iter is not None else 100
        tag = "smoke"
    else:
        dims = args.dims if args.dims is not None else [30, 50]
        functions = args.functions if args.functions is not None else cec2017.function_ids(exclude_f2=not args.include_f2)
        num_runs = args.num_runs if args.num_runs is not None else 30
        max_iter = args.max_iter if args.max_iter is not None else 1000
        tag = "full"

    for alg in args.algorithms:
        if alg not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {alg}. Valid: {sorted(ALGORITHMS)}")

    print(f"=== CEC2017 comparison: {tag} ===")
    print(f"Device    : {args.device}")
    print(f"DType     : {dtype}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Dims      : {dims}")
    print(f"Functions : {functions}")
    print(f"Runs      : {num_runs}")
    print(f"Pop size  : {args.pop_size}")
    print(f"Max iter  : {max_iter}")
    print(f"FEs/run   : {args.pop_size * max_iter}")
    print()

    os.makedirs("results", exist_ok=True)
    summary_path = f"results/cec2017_{tag}_summary.csv"
    raw_path = f"results/cec2017_{tag}_raw.csv"

    summary_rows = []
    raw_rows = []
    total_start = time.time()

    for alg_name in args.algorithms:
        print("\n==============================")
        print(f"Algorithm: {alg_name}")
        print("==============================")

        alg_log_sum = 0.0
        alg_count = 0

        for dim in dims:
            print(f"\n--- dim={dim} ---")

            for fn_id in functions:
                seed = args.seed + 100000 * len(alg_name) + 1000 * dim + fn_id

                try:
                    row, best, err = run_one(
                        alg_name=alg_name,
                        fn_id=fn_id,
                        dim=dim,
                        pop_size=args.pop_size,
                        max_iter=max_iter,
                        num_runs=num_runs,
                        seed=seed,
                        device=args.device,
                        dtype=dtype,
                    )
                except Exception as exc:
                    print(f"{alg_name:>4s} F{fn_id:02d} ERROR: {type(exc).__name__}: {exc}")
                    continue

                summary_rows.append(row)
                alg_log_sum += row["logscore"]
                alg_count += 1

                for run_idx, (b, e) in enumerate(zip(best, err)):
                    raw_rows.append({
                        "algorithm": alg_name,
                        "suite": "CEC2017",
                        "fn_id": fn_id,
                        "dim": dim,
                        "run_idx": run_idx,
                        "best_fitness": b,
                        "error": e,
                    })

                print(
                    f"{alg_name:>4s} F{fn_id:02d} | "
                    f"mean={row['mean_error']:.3e} | "
                    f"median={row['median_error']:.3e} | "
                    f"best={row['min_error']:.3e} | "
                    f"log={row['logscore']:.3f} | "
                    f"time={row['elapsed_sec']:.2f}s"
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if alg_count > 0:
            print(f"\nAlgorithm {alg_name} average logscore: {alg_log_sum / alg_count:.4f}")

    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if raw_rows:
        with open(raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)

    print("\n=== Done ===")
    print(f"Saved summary: {summary_path}")
    print(f"Saved raw    : {raw_path}")
    print(f"Total time   : {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
