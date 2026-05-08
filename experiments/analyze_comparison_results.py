import csv
from collections import defaultdict

INFILE = "results/comparison_full_summary.csv"
OUT_RANKS = "results/comparison_full_ranks.csv"
OUT_OVERVIEW = "results/comparison_algorithm_overview.csv"

rows = []
with open(INFILE, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["fn_id"] = int(r["fn_id"])
        r["dim"] = int(r["dim"])
        r["mean_error"] = float(r["mean_error"])
        r["std_error"] = float(r["std_error"])
        r["median_error"] = float(r["median_error"])
        r["logscore"] = float(r["logscore"])
        rows.append(r)

groups = defaultdict(list)
for r in rows:
    groups[(r["dim"], r["fn_id"])].append(r)

rank_rows = []
wins = defaultdict(int)
rank_sum = defaultdict(float)
rank_count = defaultdict(int)
log_sum = defaultdict(float)

for key, group in sorted(groups.items()):
    dim, fn_id = key

    # Rank by mean_error. Lower is better.
    group_sorted = sorted(group, key=lambda x: x["mean_error"])

    for rank, r in enumerate(group_sorted, start=1):
        alg = r["algorithm"]
        if rank == 1:
            wins[alg] += 1
        rank_sum[alg] += rank
        rank_count[alg] += 1
        log_sum[alg] += r["logscore"]

        out = dict(r)
        out["rank_by_mean"] = rank
        rank_rows.append(out)

with open(OUT_RANKS, "w", newline="") as f:
    fieldnames = list(rank_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rank_rows)

overview = []
for alg in sorted(rank_count.keys()):
    overview.append({
        "algorithm": alg,
        "wins_by_mean": wins[alg],
        "average_rank_by_mean": rank_sum[alg] / rank_count[alg],
        "average_logscore": log_sum[alg] / rank_count[alg],
        "num_cases": rank_count[alg],
    })

overview = sorted(overview, key=lambda x: (x["average_rank_by_mean"], x["average_logscore"]))

with open(OUT_OVERVIEW, "w", newline="") as f:
    fieldnames = ["algorithm", "wins_by_mean", "average_rank_by_mean", "average_logscore", "num_cases"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(overview)

print("=== Algorithm overview ===")
for r in overview:
    print(
        f"{r['algorithm']:>6s} | "
        f"wins={r['wins_by_mean']:2d} | "
        f"avg_rank={r['average_rank_by_mean']:.3f} | "
        f"avg_log={r['average_logscore']:.4f}"
    )

print()
print(f"Saved: {OUT_RANKS}")
print(f"Saved: {OUT_OVERVIEW}")
