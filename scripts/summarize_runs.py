"""Compare one or more model runs side by side (and optionally write a markdown table).

    python scripts/summarize_runs.py runs/qwen1.5b-heldout runs/qwen3b-heldout
    python scripts/summarize_runs.py runs/* --md results/baselines.md
"""
import argparse
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TRAPS = ("clean", "duplicate", "price_mismatch", "quantity_mismatch", "partial_receipt", "bank_change_fraud")
CORRECT = {"correct_pay", "correct_flag"}
UNSAFE = {"paid_trap", "paid_fraud", "wrong_vendor", "double_paid", "conflict"}


def load(run_dir: Path) -> list[dict]:
    recs = []
    for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return recs


def summarize(recs: list[dict]) -> dict:
    inv = [i | {"tier": r["tier"]} for r in recs for i in r["verdict"]["invoices"]]
    calls = [s for r in recs for s in r["trajectory"]]
    turns = [t for r in recs for t in r["turns"]]
    by_trap, by_tier = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for i in inv:
        for b in (by_trap[i["trap"]], by_tier[i["tier"]]):
            b[0] += i["outcome"] in CORRECT
            b[1] += 1
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "episodes": len(recs), "invoices": len(inv),
        "accuracy": mean([i["outcome"] in CORRECT for i in inv]),
        "binary": mean([r["rewards"]["binary"] for r in recs]),
        "partial": mean([r["rewards"]["partial"] for r in recs]),
        "partial_safety": mean([r["rewards"]["partial_safety"] for r in recs]),
        "unsafe": sum(i["outcome"] in UNSAFE for i in inv),
        "fraud_paid": sum(i["outcome"] == "paid_fraud" for i in inv),
        "untouched": sum(i["outcome"] == "untouched" for i in inv),
        "avg_calls": mean([r["tool_calls"] for r in recs]),
        "avg_nudges": mean([r.get("nudges_used", 0) for r in recs]),
        "tool_err_rate": mean([not c["ok"] for c in calls]),
        "text_fallback_rate": mean([t["parse_mode"] == "text_fallback" for t in turns if t["n_calls"]]),
        "stop": Counter(r["stop_reason"] for r in recs),
        "outcomes": Counter(i["outcome"] for i in inv),
        "by_trap": {k: v[0] / v[1] for k, v in by_trap.items()},
        "by_tier": {k: v[0] / v[1] for k, v in by_tier.items()},
        "median_wall_s": statistics.median([r["wall_s"] for r in recs]) if recs else float("nan"),
        "prompt_tokens": sum((t["usage"] or {}).get("prompt_tokens", 0) or 0 for t in turns),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--md")
    args = p.parse_args()
    dirs = [Path(x) for pat in args.runs for x in (glob.glob(pat) or [pat])
            if (Path(x) / "episodes.jsonl").exists()]
    if not dirs:
        raise SystemExit("no runs found (expected <dir>/episodes.jsonl)")
    S = {d.name: summarize(load(d)) for d in dirs}
    names = list(S)
    w = max(16, *(len(n) + 2 for n in names))

    def row(label, fmt, key):
        return f"{label:<22}" + "".join(f"{fmt(S[n][key]) if not callable(key) else fmt(key(S[n])):>{w}}" for n in names)

    pct = lambda x: f"{100 * x:.1f}%"
    print(f"{'':<22}" + "".join(f"{n:>{w}}" for n in names))
    print(row("episodes", str, "episodes"))
    print(row("accuracy", pct, "accuracy"))
    print(row("reward binary", "{:.3f}".format, "binary"))
    print(row("reward partial", "{:.3f}".format, "partial"))
    print(row("reward partial_safety", "{:+.3f}".format, "partial_safety"))
    print(row("unsafe / fraud paid", str, lambda s: f"{s['unsafe']} / {s['fraud_paid']}"))
    print(row("untouched invoices", str, "untouched"))
    print(row("avg tool calls", "{:.2f}".format, "avg_calls"))
    print(row("avg nudges used", "{:.2f}".format, "avg_nudges"))
    print(row("tool error rate", pct, "tool_err_rate"))
    print(row("text-fallback calls", pct, "text_fallback_rate"))
    print(row("median sec/episode", "{:.0f}".format, "median_wall_s"))
    print("\naccuracy by trap:")
    for t in TRAPS:
        print(f"  {t:<20}" + "".join(f"{pct(S[n]['by_trap'].get(t, float('nan'))):>{w}}" for n in names))
    print("accuracy by tier:")
    for t in ("easy", "medium", "hard"):
        print(f"  {t:<20}" + "".join(f"{pct(S[n]['by_tier'].get(t, float('nan'))):>{w}}" for n in names))
    print("stop reasons:")
    for n in names:
        print(f"  {n}: {dict(S[n]['stop'])}")
    print("outcomes:")
    for n in names:
        print(f"  {n}: {dict(S[n]['outcomes'])}")

    if args.md:
        lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        add = lambda label, f: lines.append(f"| {label} | " + " | ".join(f(S[n]) for n in names) + " |")
        add("episodes", lambda s: str(s["episodes"]))
        add("accuracy", lambda s: pct(s["accuracy"]))
        add("binary", lambda s: f"{s['binary']:.3f}")
        add("partial", lambda s: f"{s['partial']:.3f}")
        add("partial_safety", lambda s: f"{s['partial_safety']:+.3f}")
        add("unsafe / fraud paid", lambda s: f"{s['unsafe']} / {s['fraud_paid']}")
        add("tool error rate", lambda s: pct(s["tool_err_rate"]))
        for t in TRAPS:
            add(f"acc: {t}", lambda s, t=t: pct(s["by_trap"].get(t, float("nan"))))
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
