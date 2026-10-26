"""Summary of a scripts/eval/write_cost.mjs record: seconds per conversation, one conversation at a time.

    python3 scripts/eval/write_cost_summary.py results/eval/memory/write-cost.jsonl
"""
import json
import statistics
import sys


def quantile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main(path: str) -> None:
    lines = [json.loads(line) for line in open(path)]
    rows = [r for r in lines if "k" in r]
    body = rows[1:]  # the first conversation includes the servers' warm-up
    calls = [c for r in rows for c in r["calls"]]
    print(f"{path}: {len(rows)} conversations (the first, with the warm-up, left out below)")
    print("| per conversation | median | 90th percentile | max | mean |")
    print("|---|---|---|---|---|")
    for name, xs in [
        ("tokens", [r["tokens"] for r in body]),
        ("facts: extraction + update check (s)", [r["factsMs"] / 1000 for r in body]),
        ("embedding the message windows (s)", [r["captureMs"] / 1000 for r in body]),
        ("one extraction request (s)", [c["ms"] / 1000 for r in body for c in r["calls"] if c["name"] == "fact"]),
        ("one update-check request (s)", [c["ms"] / 1000 for r in body for c in r["calls"] if c["name"] == "updates"]),
    ]:
        print(f"| {name} | {quantile(xs, .5):.2f} | {quantile(xs, .9):.2f} | {max(xs):.2f} | {statistics.mean(xs):.2f} |")
    failed = [c for c in calls if not c["ok"]]
    print(f"requests: {len(calls)} ({sum(c['name'] == 'fact' for c in calls)} extraction, "
          f"{sum(c['name'] == 'updates' for c in calls)} update check); failed {len(failed)}, "
          f"{sum(c['ms'] for c in failed) / 1000:.1f} s of {sum(c['ms'] for c in calls) / 1000:.1f} s")
    for c in failed:
        print(f"  {c['name']}: {c['error']} ({c['ms'] / 1000:.1f} s)")
    total = sum(r["factsMs"] + r["captureMs"] for r in rows) / 1000
    print(f"all {len(rows)} conversations: {total / 60:.1f} min")
    for r in lines:
        if r.get("footer"):
            print("store:", {k: v for k, v in r.items() if k != "footer"})


if __name__ == "__main__":
    main(sys.argv[1])
