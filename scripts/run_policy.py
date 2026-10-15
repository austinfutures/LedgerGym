"""Run one scripted policy or probe through the env and show detailed verifier results.

    python scripts/run_policy.py data/tasks --policy oracle
    python scripts/run_policy.py data/tasks --policy pay_all --split heldout
    python scripts/run_policy.py data/tasks --policy hedge --show-failures 5

For the side-by-side comparison of all probes, use scripts/probe_hacks.py.
"""
import argparse

from ledgergym.evaluate import evaluate, load_manifest, summary
from ledgergym.policies import POLICIES
from ledgergym.probes import PROBES_BY_NAME
from ledgergym.rewards import REWARD_DESIGNS
from ledgergym.verifier import OUTCOMES

ALL = {**POLICIES, **{name: p.fn for name, p in PROBES_BY_NAME.items()}}


def pct(c) -> str:
    return f"{100 * c[0] / c[1]:5.1f}% (n={c[1]})" if c[1] else "n/a"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="data/tasks")
    p.add_argument("--policy", default="oracle", choices=sorted(ALL))
    p.add_argument("--split", default="all", choices=["all", "train", "heldout"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tool-calls", type=int, default=6)
    p.add_argument("--show-failures", type=int, default=10)
    args = p.parse_args()

    tasks = load_manifest(args.root, args.split, args.limit)
    agg = evaluate(tasks, ALL[args.policy], args.max_tool_calls)
    s = summary(agg)

    print(f"policy={args.policy}  split={args.split}  tasks={s['n_tasks']}  invoices={s['n_invoices']}")
    print(f"accuracy          {100 * s['accuracy']:5.1f}%")
    for k in REWARD_DESIGNS:
        print(f"reward {k:<15} {s[k]:+.3f}")
    print(f"max tool calls    {s['max_calls']}  (budget {args.max_tool_calls})")
    print(f"tool errors       {s['tool_errors']}")
    print(f"integrity fails   {s['integrity_fails']}")
    print(f"unsafe outcomes   {s['unsafe']}   (fraud paid: {s['fraud_paid']}, improper $: {s['improper_dollars']:,.2f})")
    print("\noutcomes:")
    for o in OUTCOMES:
        if agg["outcomes"][o]:
            print(f"  {o:<18} {agg['outcomes'][o]}")
    print("by trap:")
    for t in sorted(agg["by_trap"]):
        print(f"  {t:<18} {pct(agg['by_trap'][t])}")
    print("by tier:")
    for t in ("easy", "medium", "hard"):
        if t in agg["by_tier"]:
            print(f"  {t:<18} {pct(agg['by_tier'][t])}")
    if agg["crashed"]:
        print(f"\nCRASHED on {len(agg['crashed'])} task(s):")
        for tid, e in agg["crashed"][:10]:
            print(f"  {tid}: {e}")
    if agg["failures"] and args.show_failures:
        print(f"\nfirst {min(len(agg['failures']), args.show_failures)} wrong invoices:")
        for tid, inv in agg["failures"][: args.show_failures]:
            print(f"  {tid} {inv['invoice_id']}: trap={inv['trap']} -> {inv['outcome']}")


if __name__ == "__main__":
    main()
