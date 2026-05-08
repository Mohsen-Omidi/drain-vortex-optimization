from pathlib import Path
import pandas as pd
import numpy as np
import re

OUT = Path("paper_tables_with_svoa")
OUT.mkdir(parents=True, exist_ok=True)

ALG_ORDER = ["MVDO", "SVOA", "PSO", "GWO", "WOA", "SCA", "AOA", "EO"]
ALG_LABEL = {
    "MVDO": "DVO",
    "SVOA": "SVOA",
    "PSO": "PSO",
    "GWO": "GWO",
    "WOA": "WOA",
    "SCA": "SCA",
    "AOA": "AOA",
    "EO": "EO",
}

def find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def metric_col(df):
    for c in ["logscore", "log_score", "log", "avg_log", "mean_log"]:
        cc = find_col(df, [c])
        if cc:
            return cc
    cols = [c for c in df.columns if "log" in c.lower()]
    if cols:
        return cols[0]
    raise ValueError(f"No log column found. Columns: {list(df.columns)}")

def esc(s):
    s = str(s)
    s = s.replace("_", r"\_")
    return s

def fmt_float(x, digits=3):
    if pd.isna(x):
        return "--"
    x = float(x)
    return f"{x:.{digits}f}"

def fmt_sci(x):
    if pd.isna(x):
        return "--"
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"${x:.3e}$"
    return f"{x:.4g}"

def bold_if_best(txt, is_best):
    return r"\textbf{" + txt + "}" if is_best else txt

def write_table(path, caption, label, headers, rows, align=None, small=True):
    if align is None:
        align = "l" + "c" * (len(headers) - 1)

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    if small:
        lines.append(r"\scriptsize")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")

def make_logscore_table(infile, outfile, caption, label, subset_filter=None, row_prefix="F"):
    df = pd.read_csv(infile)
    alg_col = find_col(df, ["algorithm", "alg", "method"])
    f_col = find_col(df, ["function", "function_id", "fn_id", "fid", "f"])
    d_col = find_col(df, ["dim", "dimension", "d"])
    m_col = metric_col(df)

    if subset_filter:
        df = subset_filter(df, f_col, d_col)

    algs = [a for a in ALG_ORDER if a in set(df[alg_col].astype(str))]
    headers = ["Case"] + [ALG_LABEL[a] for a in algs]

    cases = df[[f_col, d_col]].drop_duplicates().sort_values([d_col, f_col])
    rows = []

    for _, c in cases.iterrows():
        f = int(c[f_col])
        d = int(c[d_col])
        sub = df[(df[f_col].astype(int) == f) & (df[d_col].astype(int) == d)]
        vals = {}
        for a in algs:
            aa = sub[sub[alg_col].astype(str) == a]
            vals[a] = float(aa[m_col].iloc[0]) if len(aa) else np.nan

        finite = [v for v in vals.values() if not pd.isna(v)]
        best = min(finite) if finite else np.nan

        row_name = f"{row_prefix}{f:02d} D{d}"
        row = [row_name]
        for a in algs:
            v = vals[a]
            txt = fmt_float(v, 3)
            row.append(bold_if_best(txt, np.isclose(v, best, rtol=1e-12, atol=1e-12)))
        rows.append(row)

    write_table(outfile, caption, label, headers, rows)

def make_engineering_table():
    infile = Path("results/with_svoa/engineering_full_summary_with_svoa.csv")
    df = pd.read_csv(infile)

    alg_col = find_col(df, ["algorithm", "alg", "method"])
    p_col = find_col(df, ["problem", "problem_name", "case"])
    m_col = find_col(df, ["best_feasible_obj"])
    fr_col = find_col(df, ["feasible_rate"])

    algs = [a for a in ALG_ORDER if a in set(df[alg_col].astype(str))]
    headers = ["Problem"] + [ALG_LABEL[a] for a in algs]
    rows = []

    for p in sorted(df[p_col].unique()):
        sub = df[df[p_col] == p]
        vals = {}
        feas = {}
        for a in algs:
            aa = sub[sub[alg_col].astype(str) == a]
            vals[a] = float(aa[m_col].iloc[0]) if len(aa) else np.nan
            feas[a] = float(aa[fr_col].iloc[0]) if len(aa) and fr_col else np.nan

        finite = [v for v in vals.values() if not pd.isna(v)]
        best = min(finite) if finite else np.nan

        row = [esc(p)]
        for a in algs:
            txt = fmt_sci(vals[a])
            if not pd.isna(feas[a]) and feas[a] < 1.0:
                txt = txt + f" ({feas[a]:.2f})"
            row.append(bold_if_best(txt, np.isclose(vals[a], best, rtol=1e-12, atol=1e-12)))
        rows.append(row)

    write_table(
        OUT / "table_engineering_best_feasible_results_with_svoa.tex",
        "Engineering design results after adding SVOA. Values are best feasible objective values over 30 runs. Values in parentheses denote feasibility rate when it is below 1.00. Lower values are better.",
        "tab:engineering_with_svoa",
        headers,
        rows,
    )

def make_friedman_table():
    analysis = Path("results/with_svoa/analysis")
    files = sorted(analysis.glob("*_friedman.csv"))
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    order = [
        ("cec2022", "overall"), ("cec2022", "D10"), ("cec2022", "D20"),
        ("cec2017", "overall"), ("cec2017", "D30"), ("cec2017", "D50"),
        ("engineering", "overall"),
        ("classical", "scalable_D30"), ("classical", "scalable_D100"),
        ("classical", "fixed_dimensional"),
    ]

    rows = []
    for suite, subset in order:
        sub = df[(df["suite"] == suite) & (df["subset"] == subset)]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        rows.append([
            esc(suite.upper()),
            esc(subset),
            str(int(r["cases"])),
            str(int(r["algorithms"])),
            f"{float(r['statistic']):.3f}",
            f"{float(r['p_value']):.3e}",
        ])

    write_table(
        OUT / "table_friedman_with_svoa.tex",
        "Friedman test results after adding SVOA. Lower ranks indicate better performance.",
        "tab:friedman_with_svoa",
        ["Suite", "Subset", "Cases", "Algorithms", "Statistic", "$p$-value"],
        rows,
        align="llcccc",
    )

def make_wilcoxon_svoa_table():
    analysis = Path("results/with_svoa/analysis")
    files = sorted(analysis.glob("*_wilcoxon.csv"))
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df = df[df["baseline"].astype(str) == "SVOA"].copy()

    order = [
        ("cec2022", "overall"), ("cec2022", "D10"), ("cec2022", "D20"),
        ("cec2017", "overall"), ("cec2017", "D30"), ("cec2017", "D50"),
        ("engineering", "overall"),
        ("classical", "scalable_D30"), ("classical", "scalable_D100"),
        ("classical", "fixed_dimensional"),
    ]

    rows = []
    for suite, subset in order:
        sub = df[(df["suite"] == suite) & (df["subset"] == subset)]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        sig = "Yes" if bool(r["significant_0.05"]) else "No"
        rows.append([
            esc(suite.upper()),
            esc(subset),
            f"{float(r['mean_log_diff']):.3f}",
            f"{float(r['p_raw']):.3e}",
            f"{float(r['p_holm']):.3e}",
            sig,
        ])

    write_table(
        OUT / "table_wilcoxon_dvo_vs_svoa.tex",
        "Pairwise Wilcoxon signed-rank comparison between DVO and SVOA. Negative mean differences favour DVO. Holm correction is applied within each suite or subset analysis.",
        "tab:wilcoxon_dvo_svoa",
        ["Suite", "Subset", "Mean diff.", "$p$", "$p_{Holm}$", "Sig."],
        rows,
        align="llcccc",
    )

# Main benchmark tables
make_logscore_table(
    "results/with_svoa/cec2022_full_summary_with_svoa.csv",
    OUT / "table_cec2022_results_with_svoa.tex",
    "CEC2022 results after adding SVOA. Values are log-scores averaged over 30 runs. Lower values are better.",
    "tab:cec2022_with_svoa",
)

make_logscore_table(
    "results/with_svoa/cec2017_full_summary_with_svoa.csv",
    OUT / "table_cec2017_D30_results_with_svoa.tex",
    "CEC2017 D30 results after adding SVOA. Values are log-scores averaged over 30 runs. Lower values are better.",
    "tab:cec2017_d30_with_svoa",
    subset_filter=lambda df, f, d: df[df[d].astype(int) == 30],
)

make_logscore_table(
    "results/with_svoa/cec2017_full_summary_with_svoa.csv",
    OUT / "table_cec2017_D50_results_with_svoa.tex",
    "CEC2017 D50 results after adding SVOA. Values are log-scores averaged over 30 runs. Lower values are better.",
    "tab:cec2017_d50_with_svoa",
    subset_filter=lambda df, f, d: df[df[d].astype(int) == 50],
)

make_logscore_table(
    "results/with_svoa/classical_full_summary_with_svoa.csv",
    OUT / "table_classical_scalable_D30_results_with_svoa.tex",
    "Classical scalable D30 benchmark results after adding SVOA. Values are log-scores averaged over 30 runs. Lower values are better.",
    "tab:classical_d30_with_svoa",
    subset_filter=lambda df, f, d: df[(df[f].astype(int) <= 13) & (df[d].astype(int) == 30)],
)

make_logscore_table(
    "results/with_svoa/classical_full_summary_with_svoa.csv",
    OUT / "table_classical_scalable_D100_results_with_svoa.tex",
    "Classical scalable D100 benchmark results after adding SVOA. Values are log-scores averaged over 30 runs. Lower values are better.",
    "tab:classical_d100_with_svoa",
    subset_filter=lambda df, f, d: df[(df[f].astype(int) <= 13) & (df[d].astype(int) == 100)],
)

make_logscore_table(
    "results/with_svoa/classical_full_summary_with_svoa.csv",
    OUT / "table_classical_fixed_dimensional_results_with_svoa.tex",
    "Classical fixed-dimensional benchmark results after adding SVOA. Values are log-scores averaged over 30 runs. Lower values are better.",
    "tab:classical_fixed_with_svoa",
    subset_filter=lambda df, f, d: df[df[f].astype(int) >= 14],
)

make_engineering_table()
make_friedman_table()
make_wilcoxon_svoa_table()

print("Saved LaTeX tables in:", OUT)
for p in sorted(OUT.glob("*.tex")):
    print(" -", p)
