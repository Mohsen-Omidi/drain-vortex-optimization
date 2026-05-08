
from __future__ import annotations
import argparse, csv, os, sys, time
from typing import Dict, Type
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.mvdo import MVDO
from algorithms.baselines import PSO, GWO, WOA, SCA, AOA, EO
from algorithms.svoa import SVOA
from benchmarks import classical

ALGORITHMS: Dict[str, Type] = {"MVDO": MVDO,
    "SVOA": SVOA, "PSO": PSO, "GWO": GWO, "WOA": WOA, "SCA": SCA, "AOA": AOA, "EO": EO}

def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "logscore": float(np.mean(np.log10(x + 1e-12))),
    }

def build_cases(mode, dims, functions):
    cases = []
    if mode == "smoke":
        chosen = functions if functions is not None else [1,3,5,8,10,12,13,14,16,19,21,23]
        for f in chosen:
            cases.append((f, dims[0] if f <= 13 else classical.native_dim(f)))
        return cases
    if mode in ["full", "scalable"]:
        chosen = functions if functions is not None else classical.scalable_function_ids()
        for d in dims:
            for f in chosen:
                if f <= 13:
                    cases.append((f, d))
    if mode in ["full", "fixed"]:
        chosen = functions if functions is not None else classical.fixed_function_ids()
        for f in chosen:
            if f >= 14:
                cases.append((f, classical.native_dim(f)))
    return cases

def run_one(alg_name, fn_id, dim, pop_size, max_iter, num_runs, seed, device, dtype):
    fn = classical.build_function(fn_id=fn_id, dim=dim, device=device, dtype=dtype)
    Alg = ALGORITHMS[alg_name]
    opt = Alg(fn=fn, dim=dim, lb=fn.lb, ub=fn.ub, pop_size=pop_size, max_iter=max_iter,
              num_runs=num_runs, device=device, dtype=dtype, seed=seed)
    start = time.time()
    result = opt.run(verbose=False)
    elapsed = time.time() - start
    best = np.asarray(result["best_fitness_per_run"], dtype=np.float64)
    err = np.maximum(best - float(fn.bias), 0.0)
    s = summarize(err)
    row = {
        "algorithm": alg_name, "suite": "Classical", "fn_id": fn_id, "dim": dim,
        "num_runs": num_runs, "pop_size": pop_size, "max_iter": max_iter,
        "fes_per_run_nominal": pop_size * max_iter,
        "mean_error": s["mean"], "std_error": s["std"], "min_error": s["min"],
        "median_error": s["median"], "max_error": s["max"], "logscore": s["logscore"],
        "mean_fitness": float(np.mean(best)), "std_fitness": float(np.std(best)),
        "min_fitness": float(np.min(best)), "median_fitness": float(np.median(best)),
        "max_fitness": float(np.max(best)), "elapsed_sec": float(elapsed),
    }
    return row, best.tolist(), err.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke","full","scalable","fixed"], default="smoke")
    parser.add_argument("--algorithms", nargs="+", default=["MVDO","PSO","GWO","WOA","SCA","AOA","EO"])
    parser.add_argument("--dims", nargs="+", type=int, default=None)
    parser.add_argument("--functions", nargs="+", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    dtype = torch.float64

    if args.mode == "smoke":
        dims = args.dims if args.dims is not None else [30]
        num_runs = args.num_runs if args.num_runs is not None else 5
        max_iter = args.max_iter if args.max_iter is not None else 100
        tag = "smoke"
    elif args.mode == "full":
        dims = args.dims if args.dims is not None else [30,100]
        num_runs = args.num_runs if args.num_runs is not None else 30
        max_iter = args.max_iter if args.max_iter is not None else 1000
        tag = "full"
    elif args.mode == "scalable":
        dims = args.dims if args.dims is not None else [30,100,500,1000]
        num_runs = args.num_runs if args.num_runs is not None else 30
        max_iter = args.max_iter if args.max_iter is not None else 1000
        tag = "scalable"
    else:
        dims = args.dims if args.dims is not None else [30]
        num_runs = args.num_runs if args.num_runs is not None else 30
        max_iter = args.max_iter if args.max_iter is not None else 1000
        tag = "fixed"

    for alg in args.algorithms:
        if alg not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {alg}. Valid: {sorted(ALGORITHMS)}")

    cases = build_cases(args.mode, dims, args.functions)
    print(f"=== Classical comparison: {tag} ===")
    print(f"Device    : {args.device}")
    print(f"DType     : {dtype}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Cases     : {cases}")
    print(f"Runs      : {num_runs}")
    print(f"Pop size  : {args.pop_size}")
    print(f"Max iter  : {max_iter}")
    print(f"FEs/run   : {args.pop_size * max_iter}\n")

    os.makedirs("results", exist_ok=True)
    summary_path = f"results/classical_{tag}_summary.csv"
    raw_path = f"results/classical_{tag}_raw.csv"
    summary_rows, raw_rows = [], []
    total_start = time.time()

    for alg_name in args.algorithms:
        print("\n==============================")
        print(f"Algorithm: {alg_name}")
        print("==============================")
        alg_log_sum, alg_count = 0.0, 0
        for fn_id, dim in cases:
            seed = args.seed + 100000*len(alg_name) + 1000*dim + fn_id
            try:
                row, best, err = run_one(alg_name, fn_id, dim, args.pop_size, max_iter, num_runs, seed, args.device, dtype)
            except Exception as exc:
                print(f"{alg_name:>4s} F{fn_id:02d} D{dim} ERROR: {type(exc).__name__}: {exc}")
                continue
            summary_rows.append(row)
            alg_log_sum += row["logscore"]; alg_count += 1
            for run_idx, (b, e) in enumerate(zip(best, err)):
                raw_rows.append({"algorithm": alg_name, "suite": "Classical", "fn_id": fn_id, "dim": dim,
                                 "run_idx": run_idx, "best_fitness": b, "error": e})
            print(f"{alg_name:>4s} F{fn_id:02d} D{dim:<4d} | mean={row['mean_error']:.3e} | "
                  f"median={row['median_error']:.3e} | best={row['min_error']:.3e} | "
                  f"log={row['logscore']:.3f} | time={row['elapsed_sec']:.2f}s")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if alg_count:
            print(f"\nAlgorithm {alg_name} average logscore: {alg_log_sum / alg_count:.4f}")

    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
    if raw_rows:
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader(); w.writerows(raw_rows)

    print("\n=== Done ===")
    print(f"Saved summary: {summary_path}")
    print(f"Saved raw    : {raw_path}")
    print(f"Total time   : {time.time()-total_start:.2f}s")

if __name__ == "__main__":
    main()
