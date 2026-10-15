"""Run every reward-hacking probe and report which reward designs each one fools.

    python scripts/probe_hacks.py data/tasks
    python scripts/probe_hacks.py data/tasks --split heldout --out results/probes_heldout.json

A design is "fooled" by a probe if the probe scores above the null policy under it.
PASS/FAIL refers to partial_safety against the probe's threshold (see ledgergym/probes.py).
"""
import argparse
import json
from pathlib import Path

from ledgergym.evaluate import evaluate, load_manifest, summary
from ledgergym.probes import PROBES, passes
from ledgergym.rewards import REWARD_DESIGNS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="data/tasks")
    p.add_argument("--split", default="all", choices=["all", "train", "heldout"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tool-calls", type=int, default=6)
    p.add_argument("--out", default="results/probes.json")
    args = p.parse_args()

    tasks = load_manifest(args.root, args.split, args.limit)
    results = {}
    for probe in PROBES:
        results[probe.name] = summary(evaluate(tasks, probe.fn, args.max_tool_calls))
    null = results["null"]

    print(f"split={args.split}  tasks={len(tasks)}\n")
    head = (f"{'probe':<16}{'what':<38}{'acc':>7}{'binary':>8}{'partial':>9}{'p+safe':>8}"
            f"{'unsafe':>8}{'fraud':>7}{'toolerr':>9}{'integ':>7}  verdict")
    print(head)
    print("-" * len(head))
    all_pass = True
    for probe in PROBES:
        s = results[probe.name]
        ok = passes(probe, s["partial_safety"]) and s["integrity_fails"] == 0 and s["crashed"] == 0
        all_pass &= ok
        s["pass"] = ok
        print(f"{probe.name:<16}{probe.what:<38}{100 * s['accuracy']:>6.1f}%{s['binary']:>8.3f}"
              f"{s['partial']:>9.3f}{s['partial_safety']:>+8.3f}{s['unsafe']:>8}{s['fraud_paid']:>7}"
              f"{s['tool_errors']:>9}{s['integrity_fails']:>7}  {'PASS' if ok else 'FAIL'}")

    print("\nprobes that score ABOVE the null policy under each design (i.e. the design is fooled):")
    for d in REWARD_DESIGNS:
        fooled = [pr.name for pr in PROBES
                  if pr.name not in ("oracle", "null") and results[pr.name][d] > null[d] + 1e-9]
        print(f"  {d:<15} {', '.join(fooled) if fooled else '(none)'}")

    print(f"\n{'ALL PROBES PASS' if all_pass else 'SOME PROBES FAIL'} under partial_safety")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "n_tasks": len(tasks), "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
