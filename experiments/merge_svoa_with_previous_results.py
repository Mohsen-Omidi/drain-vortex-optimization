from pathlib import Path
import pandas as pd

ROOT = Path("results")
OUT = ROOT / "with_svoa"
OUT.mkdir(parents=True, exist_ok=True)

jobs = [
    {
        "name": "cec2022",
        "old": ROOT / "comparison_full_summary_reference_settings.csv",
        "svoa": ROOT / "svoa_only" / "cec2022_svoa_only_summary.csv",
        "out": OUT / "cec2022_full_summary_with_svoa.csv",
    },
    {
        "name": "cec2017",
        "old": ROOT / "cec2017_reference_settings" / "cec2017_full_summary.csv",
        "svoa": ROOT / "svoa_only" / "cec2017_svoa_only_summary.csv",
        "out": OUT / "cec2017_full_summary_with_svoa.csv",
    },
    {
        "name": "classical",
        "old": ROOT / "classical_reference_settings" / "classical_full_summary.csv",
        "svoa": ROOT / "svoa_only" / "classical_svoa_only_summary.csv",
        "out": OUT / "classical_full_summary_with_svoa.csv",
    },
    {
        "name": "engineering",
        "old": ROOT / "engineering_reference_settings" / "engineering_full_summary.csv",
        "svoa": ROOT / "svoa_only" / "engineering_svoa_only_summary.csv",
        "out": OUT / "engineering_full_summary_with_svoa.csv",
    },
]

for job in jobs:
    old_path = job["old"]
    svoa_path = job["svoa"]

    if not old_path.exists():
        print(f"[MISSING OLD] {job['name']}: {old_path}")
        continue
    if not svoa_path.exists():
        print(f"[MISSING SVOA] {job['name']}: {svoa_path}")
        continue

    old = pd.read_csv(old_path)
    svoa = pd.read_csv(svoa_path)

    # Remove SVOA from old file if it is already there, to avoid duplicates.
    alg_col = None
    for c in old.columns:
        if c.lower() in ["algorithm", "alg", "method"]:
            alg_col = c
            break

    if alg_col is None:
        raise ValueError(f"No algorithm column found in {old_path}. Columns: {list(old.columns)}")

    old = old[old[alg_col].astype(str).str.upper() != "SVOA"].copy()
    merged = pd.concat([old, svoa], ignore_index=True)

    merged.to_csv(job["out"], index=False)

    print(f"[OK] {job['name']}")
    print(f"  old rows   : {len(old)}")
    print(f"  SVOA rows  : {len(svoa)}")
    print(f"  merged rows: {len(merged)}")
    print(f"  saved      : {job['out']}")

print("\nDone.")
