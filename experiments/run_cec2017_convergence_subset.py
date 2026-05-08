"""
CEC2017 convergence and stability/diversity-proxy analysis.

This script evaluates selected algorithms at several iteration checkpoints.
It is designed for convergence plots, AUC-style comparison, and final-run
stability analysis on a small but representative CEC2017 subset.

Default subset:
    F10 D50  : multimodal
    F15 D50  : hybrid
    F30 D50  : composition

Default algorithms:
    MVDO, PSO, GWO, EO

Run:
    python experiments/run_cec2017_convergence_subset.py

Optional:
    python experiments/run_cec2017_convergence_subset.py \
      --algorithms MVDO PSO GWO EO \
      --functions 10 15 30 \
      --dims 50 \
      --checkpoints 50 100 200 400 700 1000 \
      --num-runs 30
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Final/reference MVDO settings used in previous experiments.
MVDO_PARAMS = {
    "use_greedy_agent_update": True,
    "K": 6,
    "C": 0.2,
    "gamma": 0.5,
    "p_switch": 0.08,
    "adaptive_spiral": True,
    "force_splash_replace": False,
}


def load_reference_baseline_params() -> Dict[str, Dict[str, Any]]:
    """Try to load the reference-based baseline settings file if available."""
    module_candidates = [
        "baselines_reference_settings",
        "experiments.baselines_reference_settings",
        "algorithms.baselines_reference_settings",
    ]
    for mod_name in module_candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        for attr in ["BASELINE_PARAMS", "BASELINE_SETTINGS", "REFERENCE_BASELINE_PARAMS", "REFERENCE_SETTINGS"]:
            if hasattr(mod, attr):
                obj = getattr(mod, attr)
                if isinstance(obj, dict):
                    return {str(k).upper(): dict(v) for k, v in obj.items() if isinstance(v, dict)}

        # Function-style API.
        for fn_name in ["get_baseline_params", "get_params", "get_reference_params"]:
            if hasattr(mod, fn_name):
                fn = getattr(mod, fn_name)
                out = {}
                for name in ["PSO", "GWO", "WOA", "SCA", "AOA", "EO", "HHO"]:
                    try:
                        val = fn(name)
                        if isinstance(val, dict):
                            out[name] = dict(val)
                    except Exception:
                        pass
                if out:
                    return out

    return {}


REF_BASELINE_PARAMS = load_reference_baseline_params()


ALGORITHM_MODULES = {
    "MVDO": ["algorithms.mvdo"],
    "PSO": ["algorithms.pso", "algorithms.PSO", "algorithms.baselines", "algorithms.swarm"],
    "GWO": ["algorithms.gwo", "algorithms.GWO", "algorithms.baselines", "algorithms.swarm"],
    "WOA": ["algorithms.woa", "algorithms.WOA", "algorithms.baselines", "algorithms.swarm"],
    "SCA": ["algorithms.sca", "algorithms.SCA", "algorithms.baselines", "algorithms.swarm"],
    "AOA": ["algorithms.aoa", "algorithms.AOA", "algorithms.baselines", "algorithms.swarm"],
    "EO":  ["algorithms.eo",  "algorithms.EO",  "algorithms.baselines", "algorithms.swarm"],
    "HHO": ["algorithms.hho", "algorithms.HHO", "algorithms.baselines", "algorithms.swarm"],
}

ALGORITHM_CLASSES = {
    "MVDO": ["MVDO"],
    "PSO": ["PSO", "ParticleSwarmOptimization"],
    "GWO": ["GWO", "GreyWolfOptimizer", "GreyWolfOptimization"],
    "WOA": ["WOA", "WhaleOptimizationAlgorithm"],
    "SCA": ["SCA", "SineCosineAlgorithm"],
    "AOA": ["AOA", "ArithmeticOptimizationAlgorithm"],
    "EO":  ["EO", "EquilibriumOptimizer"],
    "HHO": ["HHO", "HarrisHawksOptimization", "HarrisHawksOptimizer"],
}


def find_algorithm_class(name: str):
    key = name.upper()
    module_names = ALGORITHM_MODULES.get(key, [f"algorithms.{key.lower()}"])
    class_names = ALGORITHM_CLASSES.get(key, [key])

    errors = []
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            errors.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            continue

        for cls_name in class_names:
            if hasattr(mod, cls_name):
                cls = getattr(mod, cls_name)
                if inspect.isclass(cls):
                    return cls

    raise RuntimeError(
        f"Could not find algorithm class for {name}.\n"
        f"Tried modules: {module_names}\n"
        f"Tried classes: {class_names}\n"
        "Import errors:\n" + "\n".join(errors)
    )


def filter_kwargs(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    # If constructor has **kwargs, keep everything.
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return dict(params)
    return {k: v for k, v in params.items() if k in allowed}


def get_algorithm_params(name: str) -> Dict[str, Any]:
    key = name.upper()
    if key == "MVDO":
        return dict(MVDO_PARAMS)
    return dict(REF_BASELINE_PARAMS.get(key, {}))


def _try_factory(factory: Callable, fn_id: int, dim: int, device: str, dtype: torch.dtype):
    attempts = [
        {"fn_id": fn_id, "dim": dim, "device": device, "dtype": dtype},
        {"func_id": fn_id, "dim": dim, "device": device, "dtype": dtype},
        {"func_num": fn_id, "dim": dim, "device": device, "dtype": dtype},
        {"function_id": fn_id, "dim": dim, "device": device, "dtype": dtype},
        {"fnum": fn_id, "dim": dim, "device": device, "dtype": dtype},
        {"fn_id": fn_id, "D": dim, "device": device, "dtype": dtype},
        {"func_num": fn_id, "D": dim, "device": device, "dtype": dtype},
        {"fn_id": fn_id, "dim": dim},
        {"func_id": fn_id, "dim": dim},
        {"func_num": fn_id, "dim": dim},
        {"function_id": fn_id, "dim": dim},
        {"fnum": fn_id, "dim": dim},
        {"fn_id": fn_id, "D": dim},
        {"func_num": fn_id, "D": dim},
    ]

    for kwargs in attempts:
        try:
            return factory(**kwargs)
        except TypeError:
            pass
        except Exception:
            pass

    for args in [(fn_id, dim, device, dtype), (fn_id, dim, device), (fn_id, dim)]:
        try:
            return factory(*args)
        except TypeError:
            pass
        except Exception:
            pass

    return None


def build_cec2017_function(fn_id: int, dim: int, device: str, dtype: torch.dtype):
    module_names = [
        "benchmarks.cec2017",
        "benchmarks.cec2017_funcs",
        "benchmarks.cec2017_loader",
        "benchmarks.cec2017_pytorch",
    ]
    candidate_factory_names = [
        "build_function",
        "build_problem",
        "build_cec2017_function",
        "get_function",
        "get_problem",
        "CEC2017Function",
        "CEC2017",
        "CEC2017Problem",
        "CEC2017Benchmark",
    ]

    errors = []
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            errors.append(f"{mod_name}: import failed: {type(exc).__name__}: {exc}")
            continue

        for fac_name in candidate_factory_names:
            if not hasattr(mod, fac_name):
                continue
            obj = _try_factory(getattr(mod, fac_name), fn_id, dim, device, dtype)
            if obj is not None and callable(obj):
                return obj

        for eval_name in ["evaluate", "cec2017", "cec17_test_func", "test_func"]:
            if hasattr(mod, eval_name):
                eval_fn = getattr(mod, eval_name)

                class WrappedCEC2017:
                    def __init__(self):
                        self.lb = torch.full((dim,), -100.0, device=device, dtype=dtype)
                        self.ub = torch.full((dim,), 100.0, device=device, dtype=dtype)
                        self.bias = 100.0 * fn_id
                        self.dim = dim

                    def __call__(self, x: torch.Tensor) -> torch.Tensor:
                        try:
                            return eval_fn(x, fn_id)
                        except TypeError:
                            try:
                                return eval_fn(x, func_num=fn_id)
                            except TypeError:
                                return eval_fn(fn_id, x)

                return WrappedCEC2017()

        visible = [x for x in dir(mod) if not x.startswith("_")]
        errors.append(f"{mod_name}: no usable factory found. Visible symbols: {visible[:50]}")

    raise RuntimeError(
        "Could not build CEC2017 function from local wrappers.\n"
        "Please send the first 120 lines of experiments/run_cec2017_comparison.py "
        "and benchmarks/cec2017.py so this builder can be matched.\n\n"
        + "\n".join(errors)
    )


def get_bounds(fn, dim: int, device: str, dtype: torch.dtype):
    lb = getattr(fn, "lb", None)
    ub = getattr(fn, "ub", None)
    if lb is None:
        lb = torch.full((dim,), -100.0, device=device, dtype=dtype)
    else:
        lb = torch.as_tensor(lb, device=device, dtype=dtype)
    if ub is None:
        ub = torch.full((dim,), 100.0, device=device, dtype=dtype)
    else:
        ub = torch.as_tensor(ub, device=device, dtype=dtype)
    return lb, ub


def get_bias(fn, fn_id: int) -> float:
    for name in ["bias", "optimum", "f_opt", "fopt", "global_optimum"]:
        if hasattr(fn, name):
            try:
                return float(getattr(fn, name))
            except Exception:
                pass
    return float(100.0 * fn_id)


def extract_best_fitness(result: Dict[str, Any]) -> np.ndarray:
    keys = [
        "best_fitness_per_run",
        "best_fitness",
        "best_scores",
        "best_values",
        "gbest_fitness_per_run",
        "gbest_fitness",
    ]
    for k in keys:
        if k in result:
            arr = np.asarray(result[k], dtype=np.float64)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            return arr

    raise KeyError(
        "Could not find best fitness in optimizer result. "
        f"Available keys: {list(result.keys())}"
    )


def extract_positions(result: Dict[str, Any]) -> Optional[np.ndarray]:
    keys = [
        "best_position_per_run",
        "best_positions_per_run",
        "best_positions",
        "best_solution_per_run",
        "best_solutions_per_run",
        "best_solutions",
        "gbest_position_per_run",
        "gbest_positions",
    ]
    for k in keys:
        if k in result:
            arr = np.asarray(result[k], dtype=np.float64)
            if arr.ndim == 2:
                return arr
    return None


def normalized_solution_diversity(positions: Optional[np.ndarray], lb: torch.Tensor, ub: torch.Tensor) -> float:
    """Mean pairwise distance among final best solutions, normalized by search diameter."""
    if positions is None:
        return float("nan")
    if positions.ndim != 2 or positions.shape[0] < 2:
        return float("nan")

    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    iu = np.triu_indices(positions.shape[0], k=1)
    mean_dist = float(np.mean(dists[iu]))

    diameter = float(torch.linalg.norm(ub - lb).detach().cpu().item())
    if diameter <= 0:
        return float("nan")
    return mean_dist / diameter


def run_checkpoint(
    algorithm: str,
    fn_id: int,
    dim: int,
    max_iter: int,
    pop_size: int,
    num_runs: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
):
    cls = find_algorithm_class(algorithm)
    base_params = get_algorithm_params(algorithm)
    params = filter_kwargs(cls, base_params)

    fn = build_cec2017_function(fn_id, dim, device, dtype)
    lb, ub = get_bounds(fn, dim, device, dtype)
    bias = get_bias(fn, fn_id)

    common = {
        "fn": fn,
        "dim": dim,
        "lb": lb,
        "ub": ub,
        "pop_size": pop_size,
        "max_iter": max_iter,
        "num_runs": num_runs,
        "device": device,
        "dtype": dtype,
        "seed": seed,
    }
    common = filter_kwargs(cls, common)

    opt = cls(**common, **params)
    start = time.time()
    result = opt.run(verbose=False)
    elapsed = time.time() - start

    best = extract_best_fitness(result)
    err = np.maximum(best - bias, 0.0)
    log_err = np.log10(err + 1e-12)
    positions = extract_positions(result)
    sol_div = normalized_solution_diversity(positions, lb, ub)

    summary = {
        "algorithm": algorithm.upper(),
        "fn_id": fn_id,
        "dim": dim,
        "max_iter": max_iter,
        "fes_per_run": pop_size * max_iter,
        "num_runs": num_runs,
        "pop_size": pop_size,
        "seed": seed,
        "bias": bias,
        "params_used": repr(params),
        "mean_error": float(np.mean(err)),
        "median_error": float(np.median(err)),
        "best_error": float(np.min(err)),
        "std_error": float(np.std(err)),
        "mean_log_error": float(np.mean(log_err)),
        "std_log_error": float(np.std(log_err)),
        "min_log_error": float(np.min(log_err)),
        "median_log_error": float(np.median(log_err)),
        "solution_diversity_norm": sol_div,
        "elapsed_sec": float(elapsed),
    }

    raw = []
    for i, (b, e, le) in enumerate(zip(best, err, log_err)):
        raw.append({
            "algorithm": algorithm.upper(),
            "fn_id": fn_id,
            "dim": dim,
            "max_iter": max_iter,
            "fes_per_run": pop_size * max_iter,
            "run_idx": i,
            "best_fitness": float(b),
            "error": float(e),
            "log_error": float(le),
        })

    return summary, raw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithms", nargs="+", default=["MVDO", "PSO", "GWO", "EO"])
    p.add_argument("--functions", nargs="+", type=int, default=[10, 15, 30])
    p.add_argument("--dims", nargs="+", type=int, default=[50])
    p.add_argument("--checkpoints", nargs="+", type=int, default=[50, 100, 200, 400, 700, 1000])
    p.add_argument("--num-runs", type=int, default=30)
    p.add_argument("--pop-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.float64

    os.makedirs("results", exist_ok=True)
    summary_path = "results/convergence_cec2017_subset_summary.csv"
    raw_path = "results/convergence_cec2017_subset_raw.csv"

    print("=== CEC2017 convergence/stability subset ===")
    print(f"Device     : {args.device}")
    print(f"DType      : {dtype}")
    print(f"Algorithms : {[a.upper() for a in args.algorithms]}")
    print(f"Functions  : {args.functions}")
    print(f"Dims       : {args.dims}")
    print(f"Checkpoints: {args.checkpoints}")
    print(f"Runs       : {args.num_runs}")
    print(f"Pop size   : {args.pop_size}")
    print()

    if REF_BASELINE_PARAMS:
        print("Reference baseline settings loaded.")
    else:
        print("Using reference-based baseline defaults from algorithms/baselines.py.")
    print()

    summary_rows = []
    raw_rows = []
    total_start = time.time()

    cases = [(fn, dim) for dim in args.dims for fn in args.functions]
    for algorithm in args.algorithms:
        algorithm = algorithm.upper()
        print("\n==============================")
        print(f"Algorithm: {algorithm}")
        print("==============================")

        for fn_id, dim in cases:
            print(f"\n--- F{fn_id:02d} D{dim} ---")
            for max_iter in args.checkpoints:
                alg_index = args.algorithms.index(algorithm) if algorithm in args.algorithms else 0
                seed = args.seed + 100000 * (alg_index + 1) + 1000 * dim + 10 * fn_id + max_iter

                row, raw = run_checkpoint(
                    algorithm=algorithm,
                    fn_id=fn_id,
                    dim=dim,
                    max_iter=max_iter,
                    pop_size=args.pop_size,
                    num_runs=args.num_runs,
                    seed=seed,
                    device=args.device,
                    dtype=dtype,
                )

                summary_rows.append(row)
                raw_rows.extend(raw)

                print(
                    f"{algorithm:<5s} iter={max_iter:<4d} FE={row['fes_per_run']:<6d} | "
                    f"mean={row['mean_error']:.3e} | "
                    f"median={row['median_error']:.3e} | "
                    f"best={row['best_error']:.3e} | "
                    f"log={row['mean_log_error']:.3f} | "
                    f"stdlog={row['std_log_error']:.3f} | "
                    f"div={row['solution_diversity_norm']:.3e} | "
                    f"time={row['elapsed_sec']:.2f}s"
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

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
