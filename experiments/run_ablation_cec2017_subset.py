"""
CEC2017 subset ablation study for MVDO.

Main subset:
    F1, F3     : unimodal / basic shifted-rotated functions
    F5, F10    : multimodal functions
    F15, F20   : hybrid functions
    F25, F30   : composition functions
    D = 30, 50

Smoke:
    python experiments/run_ablation_cec2017_subset.py --mode smoke

Full subset:
    python experiments/run_ablation_cec2017_subset.py --mode full

The script is intentionally independent from the baseline algorithms and only
compares MVDO variants. It tries to use the existing local CEC2017 benchmark
wrapper from benchmarks/cec2017.py.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import importlib
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.mvdo import MVDO


FULL_MVDO_PARAMS = {
    "use_greedy_agent_update": True,
    "K": 6,
    "C": 0.2,
    "gamma": 0.5,
    "p_switch": 0.08,
    "adaptive_spiral": True,
    "force_splash_replace": False,
}


VARIANT_PARAMS: Dict[str, Dict[str, Any]] = {
    # Final/reference MVDO.
    "MVDO_full": dict(FULL_MVDO_PARAMS),

    # Remove greedy acceptance/update. This tests whether elitist retention is
    # driving performance.
    "no_greedy": {
        **FULL_MVDO_PARAMS,
        "use_greedy_agent_update": False,
    },

    # Remove probabilistic switching between vortex basins.
    "no_switch": {
        **FULL_MVDO_PARAMS,
        "p_switch": 0.0,
    },

    # Collapse the multi-vortex structure to a single basin.
    "single_vortex": {
        **FULL_MVDO_PARAMS,
        "K": 1,
        "p_switch": 0.0,
    },

    # Remove the tangential/free-vortex swirl term. The algorithm becomes mostly
    # radial drain/exploitation plus stochastic exploration.
    "no_swirl": {
        **FULL_MVDO_PARAMS,
        "C": 0.0,
    },

    # Use the earlier non-adaptive spiral update.
    "no_adaptive_spiral": {
        **FULL_MVDO_PARAMS,
        "adaptive_spiral": False,
    },

    # Disable splash-out if the current MVDO implementation exposes p_splash.
    # If p_splash is not in MVDO.__init__, this parameter is safely ignored.
    "no_splash": {
        **FULL_MVDO_PARAMS,
        "p_splash": 0.0,
        "force_splash_replace": False,
    },

    # A stricter radial-only version: no switching and no swirl.
    "radial_only": {
        **FULL_MVDO_PARAMS,
        "C": 0.0,
        "p_switch": 0.0,
    },
}


def filter_mvdo_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only parameters accepted by the installed MVDO class."""
    sig = inspect.signature(MVDO.__init__)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in params.items() if k in allowed}


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
            # It accepted the signature but failed internally; still try others.
            pass

    attempts_positional = [
        (fn_id, dim, device, dtype),
        (fn_id, dim, device),
        (fn_id, dim),
    ]
    for args in attempts_positional:
        try:
            return factory(*args)
        except TypeError:
            pass
        except Exception:
            pass

    return None


def build_cec2017_function(fn_id: int, dim: int, device: str, dtype: torch.dtype):
    """Try several common local wrapper APIs and return a callable benchmark."""
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
            factory = getattr(mod, fac_name)
            obj = _try_factory(factory, fn_id, dim, device, dtype)
            if obj is not None and callable(obj):
                return obj

        # Some wrappers expose a module-level evaluate function. Wrap it.
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

    msg = "\n".join(errors)
    raise RuntimeError(
        "Could not build CEC2017 function from local wrappers.\n"
        "Please send the first 120 lines of experiments/run_cec2017_comparison.py "
        "and benchmarks/cec2017.py so this builder can be matched.\n\n"
        + msg
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
    # CEC2017 official biases are usually 100*f_i.
    return float(100.0 * fn_id)


def summarize(x: np.ndarray) -> Dict[str, float]:
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
    variant_name: str,
    variant_params: Dict[str, Any],
    fn_id: int,
    dim: int,
    pop_size: int,
    max_iter: int,
    num_runs: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
):
    fn = build_cec2017_function(fn_id, dim, device, dtype)
    lb, ub = get_bounds(fn, dim, device, dtype)
    bias = get_bias(fn, fn_id)

    kwargs = filter_mvdo_kwargs(variant_params)

    opt = MVDO(
        fn=fn,
        dim=dim,
        lb=lb,
        ub=ub,
        pop_size=pop_size,
        max_iter=max_iter,
        num_runs=num_runs,
        device=device,
        dtype=dtype,
        seed=seed,
        **kwargs,
    )

    start = time.time()
    result = opt.run(verbose=False)
    elapsed = time.time() - start

    best = np.asarray(result["best_fitness_per_run"], dtype=np.float64)
    err = np.maximum(best - bias, 0.0)
    s = summarize(err)

    row = {
        "variant": variant_name,
        "suite": "CEC2017_ablation_subset",
        "fn_id": fn_id,
        "dim": dim,
        "num_runs": num_runs,
        "pop_size": pop_size,
        "max_iter": max_iter,
        "fes_per_run_nominal": pop_size * max_iter,
        "params_used": repr(kwargs),
        "bias": bias,
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

    raw_rows = []
    for run_idx, (b, e) in enumerate(zip(best, err)):
        raw_rows.append({
            "variant": variant_name,
            "suite": "CEC2017_ablation_subset",
            "fn_id": fn_id,
            "dim": dim,
            "run_idx": run_idx,
            "best_fitness": float(b),
            "error": float(e),
        })

    return row, raw_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_PARAMS.keys()))
    parser.add_argument("--dims", nargs="+", type=int, default=None)
    parser.add_argument("--functions", nargs="+", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64

    if args.mode == "smoke":
        dims = args.dims if args.dims is not None else [30]
        functions = args.functions if args.functions is not None else [1, 3, 10, 20, 30]
        num_runs = args.num_runs if args.num_runs is not None else 5
        max_iter = args.max_iter if args.max_iter is not None else 100
        tag = "smoke"
    else:
        dims = args.dims if args.dims is not None else [30, 50]
        functions = args.functions if args.functions is not None else [1, 3, 5, 10, 15, 20, 25, 30]
        num_runs = args.num_runs if args.num_runs is not None else 30
        max_iter = args.max_iter if args.max_iter is not None else 1000
        tag = "full"

    for v in args.variants:
        if v not in VARIANT_PARAMS:
            raise ValueError(f"Unknown variant {v}. Valid variants: {list(VARIANT_PARAMS)}")

    cases = [(fn, dim) for dim in dims for fn in functions]

    print(f"=== MVDO CEC2017 ablation subset: {tag} ===")
    print(f"Device   : {args.device}")
    print(f"DType    : {dtype}")
    print(f"Variants : {args.variants}")
    print(f"Dims     : {dims}")
    print(f"Functions: {functions}")
    print(f"Cases    : {len(cases)}")
    print(f"Runs     : {num_runs}")
    print(f"Pop size : {args.pop_size}")
    print(f"Max iter : {max_iter}")
    print(f"FEs/run  : {args.pop_size * max_iter}")
    print()

    os.makedirs("results", exist_ok=True)
    summary_path = f"results/ablation_cec2017_subset_{tag}_summary.csv"
    raw_path = f"results/ablation_cec2017_subset_{tag}_raw.csv"

    summary_rows = []
    raw_rows = []
    total_start = time.time()

    for variant_name in args.variants:
        params = VARIANT_PARAMS[variant_name]
        params_used = filter_mvdo_kwargs(params)

        print("\n==============================")
        print(f"Variant: {variant_name}")
        print(f"Params : {params_used}")
        print("==============================")

        variant_log_sum = 0.0
        variant_count = 0

        for fn_id, dim in cases:
            seed = args.seed + 100000 * (list(VARIANT_PARAMS.keys()).index(variant_name) + 1) + 1000 * dim + fn_id

            row, rr = run_one(
                variant_name=variant_name,
                variant_params=params,
                fn_id=fn_id,
                dim=dim,
                pop_size=args.pop_size,
                max_iter=max_iter,
                num_runs=num_runs,
                seed=seed,
                device=args.device,
                dtype=dtype,
            )

            summary_rows.append(row)
            raw_rows.extend(rr)
            variant_log_sum += row["logscore"]
            variant_count += 1

            print(
                f"{variant_name:<18s} F{fn_id:02d} D{dim:<3d} | "
                f"mean={row['mean_error']:.3e} | "
                f"median={row['median_error']:.3e} | "
                f"best={row['min_error']:.3e} | "
                f"log={row['logscore']:.3f} | "
                f"time={row['elapsed_sec']:.2f}s"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if variant_count > 0:
            print(f"\nVariant {variant_name} average logscore: {variant_log_sum / variant_count:.4f}")

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
