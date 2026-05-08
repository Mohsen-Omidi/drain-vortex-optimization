from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

INFILES = {
    "cec2022": Path("results/with_svoa/cec2022_full_summary_with_svoa.csv"),
    "cec2017": Path("results/with_svoa/cec2017_full_summary_with_svoa.csv"),
    "classical": Path("results/with_svoa/classical_full_summary_with_svoa.csv"),
    "engineering": Path("results/with_svoa/engineering_full_summary_with_svoa.csv"),
}

OUTDIR = Path("results/with_svoa/analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

def find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def find_log_col(df):
    candidates = [
        "ranking_score", "logscore", "log_score", "log", "avg_log", "mean_log",
        "log_error", "score_log"
    ]
    c = find_col(df, candidates)
    if c is not None:
        return c
    log_cols = [c for c in df.columns if "log" in c.lower()]
    if log_cols:
        return log_cols[0]
    raise ValueError(f"No log/score column found. Columns: {list(df.columns)}")

def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted

def detect_case_cols(df, suite):
    cols = [c.lower() for c in df.columns]
    original = {c.lower(): c for c in df.columns}

    if suite == "engineering":
        for name in ["problem", "problem_name", "case", "function"]:
            if name in original:
                return [original[name]]

    fcol = None
    for name in ["function", "function_id", "fn_id", "fid", "f"]:
        if name in original:
            fcol = original[name]
            break

    dcol = None
    for name in ["dim", "dimension", "d"]:
        if name in original:
            dcol = original[name]
            break

    if fcol and dcol:
        return [fcol, dcol]
    if fcol:
        return [fcol]

    for name in ["case", "case_id", "problem"]:
        if name in original:
            return [original[name]]

    raise ValueError(f"Could not detect case columns. Columns: {list(df.columns)}")

def add_subset(df, suite):
    df = df.copy()
    if suite == "cec2022":
        dcol = find_col(df, ["dim", "dimension", "d"])
        df["subset"] = "overall"
        if dcol:
            df.loc[df[dcol].astype(int) == 10, "subset"] = "D10"
            df.loc[df[dcol].astype(int) == 20, "subset"] = "D20"

    elif suite == "cec2017":
        dcol = find_col(df, ["dim", "dimension", "d"])
        df["subset"] = "overall"
        if dcol:
            df.loc[df[dcol].astype(int) == 30, "subset"] = "D30"
            df.loc[df[dcol].astype(int) == 50, "subset"] = "D50"

    elif suite == "classical":
        fcol = find_col(df, ["function", "function_id", "fn_id", "fid", "f"])
        dcol = find_col(df, ["dim", "dimension", "d"])
        df["subset"] = "overall"
        if fcol and dcol:
            f = df[fcol].astype(int)
            d = df[dcol].astype(int)
            df["subset"] = "fixed_dimensional"
            df.loc[(f <= 13) & (d == 30), "subset"] = "scalable_D30"
            df.loc[(f <= 13) & (d == 100), "subset"] = "scalable_D100"
            df.loc[(f <= 13) & (d.isin([30, 100])), "subset_all"] = "scalable_all"
        else:
            df["subset"] = "overall"

    else:
        df["subset"] = "overall"

    return df

def analyze_one(df, suite, subset_name, suffix=""):
    alg_col = find_col(df, ["algorithm", "alg", "method"])
    metric_col = find_log_col(df)
    case_cols = detect_case_cols(df, suite)

    if alg_col is None:
        raise ValueError(f"No algorithm column found. Columns: {list(df.columns)}")

    df = df.copy()
    df[alg_col] = df[alg_col].astype(str)

    # Pivot by case. Lower metric is better.
    wide = df.pivot_table(index=case_cols, columns=alg_col, values=metric_col, aggfunc="first")
    wide = wide.dropna(axis=0, how="any")

    algs = list(wide.columns)
    target = "MVDO" if "MVDO" in algs else ("DVO" if "DVO" in algs else None)
    if target is None:
        print(f"[WARN] No MVDO/DVO found in {suite} {subset_name}")
        return

    ranks = wide.rank(axis=1, method="average", ascending=True)

    wins = {}
    for alg in algs:
        wins[alg] = int((wide[alg] == wide.min(axis=1)).sum())

    overview = pd.DataFrame({
        "algorithm": algs,
        "wins": [wins[a] for a in algs],
        "avg_rank": [ranks[a].mean() for a in algs],
        "avg_log": [wide[a].mean() for a in algs],
    }).sort_values(["avg_rank", "avg_log"])

    # Friedman
    stat, p_friedman = friedmanchisquare(*[wide[a].values for a in algs])
    friedman_df = pd.DataFrame([{
        "suite": suite,
        "subset": subset_name,
        "cases": len(wide),
        "algorithms": len(algs),
        "statistic": stat,
        "p_value": p_friedman,
    }])

    # Wilcoxon target vs all baselines
    rows = []
    pvals = []
    baselines = [a for a in algs if a != target]
    for b in baselines:
        diff = wide[target] - wide[b]
        try:
            res = wilcoxon(wide[target], wide[b], alternative="less", zero_method="wilcox")
            p_raw = float(res.pvalue)
        except ValueError:
            p_raw = 1.0
        pvals.append(p_raw)
        rows.append({
            "suite": suite,
            "subset": subset_name,
            "comparison": f"{target} vs {b}",
            "baseline": b,
            "p_raw": p_raw,
            "mean_log_diff": float(diff.mean()),
        })

    if rows:
        p_holm = holm_adjust(pvals)
        for r, ph in zip(rows, p_holm):
            r["p_holm"] = ph
            r["significant_0.05"] = bool(ph < 0.05)
    wilcoxon_df = pd.DataFrame(rows).sort_values("p_raw")

    stem = f"{suite}_{subset_name}{suffix}".replace("/", "_")
    overview.to_csv(OUTDIR / f"{stem}_overview.csv", index=False)
    ranks.reset_index().to_csv(OUTDIR / f"{stem}_ranks.csv", index=False)
    friedman_df.to_csv(OUTDIR / f"{stem}_friedman.csv", index=False)
    wilcoxon_df.to_csv(OUTDIR / f"{stem}_wilcoxon.csv", index=False)

    print("\n" + "=" * 70)
    print(f"{suite} | subset: {subset_name} | cases: {len(wide)}")
    print("=" * 70)
    print(overview.to_string(index=False))
    print("\nFriedman:")
    print(f"statistic = {stat:.6f}")
    print(f"p-value   = {p_friedman:.6e}")
    print("\nWilcoxon: MVDO/DVO vs baselines, alternative='less'")
    if not wilcoxon_df.empty:
        print(wilcoxon_df.to_string(index=False))

for suite, path in INFILES.items():
    if not path.exists():
        print(f"[MISSING] {path}")
        continue

    df = pd.read_csv(path)
    df = add_subset(df, suite)

    analyze_one(df, suite, "overall")

    if suite in ["cec2022", "cec2017"]:
        for sub in sorted([s for s in df["subset"].unique() if s != "overall"]):
            analyze_one(df[df["subset"] == sub], suite, sub)

    if suite == "classical":
        # Main separated reporting
        for sub in ["scalable_D30", "scalable_D100", "fixed_dimensional"]:
            analyze_one(df[df["subset"] == sub], suite, sub)

        # Optional scalable_all
        if "subset_all" in df.columns:
            analyze_one(df[df["subset_all"] == "scalable_all"], suite, "scalable_all")

print("\nSaved all outputs in:", OUTDIR)
