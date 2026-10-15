"""Summarize trap / herring / action distributions over a generated task set.

    python scripts/stats.py data/tasks
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/tasks")
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]

    traps = defaultdict(Counter)
    herrings = defaultdict(Counter)
    actions = defaultdict(Counter)
    for r in rows:
        gt = json.loads((root / r["path"] / "ground_truth.json").read_text())
        key = (r["tier"], r["split"])
        for inv in gt["invoices"]:
            traps[key][inv["trap"]] += 1
            actions[key][inv["expected_action"]] += 1
            for h in inv["herrings"]:
                herrings[key][h] += 1
        for b in gt["background_herrings"]:
            herrings[key][b["herring"] + " (bg)"] += 1

    all_traps = ["clean", "duplicate", "price_mismatch", "quantity_mismatch", "partial_receipt", "bank_change_fraud"]
    print(f"{'tier/split':<16}" + "".join(f"{t[:12]:>14}" for t in all_traps) + f"{'pay':>7}{'flag':>7}")
    for key in sorted(traps):
        print(f"{key[0] + '/' + key[1]:<16}" + "".join(f"{traps[key][t]:>14}" for t in all_traps)
              + f"{actions[key]['pay']:>7}{actions[key]['flag']:>7}")
    print("\nred herrings:")
    for key in sorted(herrings):
        print(f"  {key[0] + '/' + key[1]:<16} {dict(herrings[key])}")


if __name__ == "__main__":
    main()
