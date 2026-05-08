"""
Run MVDO and baseline optimizers on CEC2022.

Usage examples on the cluster:

    # Fast smoke test before the full run
    python experiments/run_cec2022_comparison.py --mode smoke

    # Full benchmark used for tables
    python experiments/run_cec2022_comparison.py --mode full

    # Run only a subset
    python experiments/run_cec2022_comparison.py --mode full --algorithms MVDO PSO GWO WOA

Outputs:
    results/comparison_<mode>_summary.csv
    results/comparison_<mode>_raw.csv
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import torch

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import cec2022
from benchmarks.cec2022 import set_data_dir
from algorithms.mvdo import MVDO
from algorithms.baselines import PSO, GWO, WOA, SCA, HHO, AOA, EO
from algorithms.svoa import SVOA


DATA_DIR = "data/cec2022_data"
DEVICE_DEFAULT = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

ALGORITHMS = {
    "MVDO": MVDO,
    "SVOA": SVOA,
    "PSO": PSO,
    "GWO": GWO,
    "WOA": WOA,
    "SCA": SCA,
    "HHO": HHO,
    "AOA": AOA,
    "EO": EO,
}

# Different algorithms consume different numbers of random tensors per iteration.
# This seed offset makes repeated algorithm names reproducible while reducing
# accidental identical starting populations across algorithms.
ALG_SEED_OFFSET = {
    'SVOA': 7000,
    "MVDO": 1000,
    "PSO": 2000,
    "GWO": 3000,
    "WOA": 4000,
    "SCA": 5000,
    "HHO": 6000,
    "AOA": 7000,
    "EO": 8000,
}


def parse_list_int(values):
    if values is None:
        return None
    out = []
    for v in values:
        for part in str(v).split(','):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def summarize(err):
    err = np.asarray(err, dtype=np.float64)
    return {
        "mean_error": float(np.mean(err)),
        "std_error": float(np.std(err)),
        "min_error": float(np.min(err)),
        "median_error": float(np.median(err)),
        "max_error": float(np.max(err)),
        "logscore": float(np.mean(np.log10(err + 1e-12))),
    }


def mode_defaults(mode):
    if mode == "smoke":
        return {
            "dims": [10, 20],
            "functions": [1, 2, 5, 6, 10, 12],
            "num_runs": 5,
            "pop_size": 30,
            "max_iter": 100,
        }
    if mode == "pilot":
        return {
            "dims": [10, 20],
            "functions": [1, 2, 3, 5, 6, 8, 9, 10, 11, 12],
            "num_runs": 10,
            "pop_size": 30,
            "max_iter": 500,
        }
    if mode == "full":
        return {
            "dims": [10, 20],
            "functions": list(range(1, 13)),
            "num_runs": 30,
            "pop_size": 30,
            "max_iter": 1000,
        }
    raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--algorithms", nargs="*", default=list(ALGORITHMS.keys()),
                        help="Algorithm names, e.g. MVDO PSO GWO WOA SCA HHO AOA EO")
    parser.add_argument("--dims", nargs="*", default=None, help="Dimensions, e.g. 10 20")
    parser.add_argument("--functions", nargs="*", default=None, help="Function IDs, e.g. 1 2 3 or 1,2,3")
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    defaults = mode_defaults(args.mode)
    dims = parse_list_int(args.dims) or defaults["dims"]
    functions = parse_list_int(args.functions) or defaults["functions"]
    num_runs = args.num_runs or defaults["num_runs"]
    pop_size = args.pop_size or defaults["pop_size"]
    max_iter = args.max_iter or defaults["max_iter"]
    algorithms = args.algorithms

    for alg in algorithms:
        if alg not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {alg}. Available: {list(ALGORITHMS.keys())}")

    set_data_dir(DATA_DIR)
    os.makedirs("results", exist_ok=True)
    summary_csv = f"results/comparison_{args.mode}_summary.csv"
    raw_csv = f"results/comparison_{args.mode}_raw.csv"

    print(f"=== CEC2022 comparison: {args.mode} ===")
    print(f"Device    : {args.device}")
    print(f"DType     : {DTYPE}")
    print(f"Data dir  : {DATA_DIR}")
    print(f"Algorithms: {algorithms}")
    print(f"Dims      : {dims}")
    print(f"Functions : {functions}")
    print(f"Runs      : {num_runs}")
    print(f"Pop size  : {pop_size}")
    print(f"Max iter  : {max_iter}")
    print(f"FEs/run   : {pop_size * max_iter}")
    print()

    summary_rows = []
    raw_rows = []
    total_start = time.time()

    for alg_name in algorithms:
        AlgClass = ALGORITHMS[alg_name]
        print("\n==============================")
        print(f"Algorithm: {alg_name}")
        print("==============================")

        alg_logscore_total = 0.0
        alg_count = 0

        for dim in dims:
            print(f"\n--- dim={dim} ---")
            for fn_id in functions:
                fn = cec2022.build_function(
                    fn_id=fn_id,
                    dim=dim,
                    device=args.device,
                    dtype=DTYPE,
                )

                seed = args.seed + ALG_SEED_OFFSET[alg_name] + 1000 * dim + fn_id
                opt = AlgClass(
                    fn=fn,
                    dim=dim,
                    lb=fn.lb,
                    ub=fn.ub,
                    pop_size=pop_size,
                    max_iter=max_iter,
                    num_runs=num_runs,
                    device=args.device,
                    dtype=DTYPE,
                    seed=seed,
                )

                start = time.time()
                result = opt.run(verbose=False)
                elapsed = time.time() - start

                best = np.asarray(result["best_fitness_per_run"], dtype=np.float64)
                err = np.maximum(best - float(fn.bias), 0.0)
                s = summarize(err)
                alg_logscore_total += s["logscore"]
                alg_count += 1

                row = {
                    "algorithm": alg_name,
                    "fn_id": fn_id,
                    "dim": dim,
                    "num_runs": num_runs,
                    "pop_size": pop_size,
                    "max_iter": max_iter,
                    "fes_per_run": pop_size * max_iter,
                    **s,
                    "elapsed_sec": float(elapsed),
                }
                summary_rows.append(row)

                for run_idx, (fit_val, err_val) in enumerate(zip(best, err)):
                    raw_rows.append({
                        "algorithm": alg_name,
                        "fn_id": fn_id,
                        "dim": dim,
                        "run": run_idx,
                        "best_fitness": float(fit_val),
                        "error": float(err_val),
                    })

                print(
                    f"{alg_name:>4s} F{fn_id:02d} | "
                    f"mean={s['mean_error']:.3e} | "
                    f"median={s['median_error']:.3e} | "
                    f"best={s['min_error']:.3e} | "
                    f"log={s['logscore']:.3f} | "
                    f"time={elapsed:.2f}s"
                )

                del opt, result, best, err, fn
                if torch.cuda.is_available() and "cuda" in args.device:
                    torch.cuda.empty_cache()

        avg_log = alg_logscore_total / max(1, alg_count)
        print(f"\nAlgorithm {alg_name} average logscore: {avg_log:.4f}")

        # Save incrementally after each algorithm to avoid losing results if a long run stops.
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        with open(raw_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)

    total_elapsed = time.time() - total_start
    print("\n=== Done ===")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved raw    : {raw_csv}")
    print(f"Total time   : {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
