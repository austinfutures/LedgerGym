"""Generate the LedgerGym task set with a stratified train/held-out split.

    python scripts/generate_tasks.py --out data/tasks --n 500 --seed 0
"""
import argparse
import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path

from ledgergym.generator import TIERS, TRAPS, generate_task


def task_seed(base: int, i: int) -> int:
    return int(hashlib.sha256(f"ledgergym:{base}:{i}".encode()).hexdigest()[:12], 16)


def allocate(n: int, mix: dict[str, float]) -> dict[str, int]:
    total = sum(mix.values())
    raw = {t: n * w / total for t, w in mix.items()}
    counts = {t: int(r) for t, r in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - counts[t], reverse=True)[: n - sum(counts.values())]:
        counts[t] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tasks")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heldout-frac", type=float, default=0.2)
    p.add_argument("--mix", default="easy:0.4,medium:0.35,hard:0.25")
    args = p.parse_args()

    mix = {k: float(v) for k, v in (kv.split(":") for kv in args.mix.split(","))}
    unknown = set(mix) - set(TIERS)
    if unknown:
        raise SystemExit(f"unknown tiers: {unknown}")
    counts = allocate(args.n, mix)

    rng = random.Random(args.seed)
    tiers = [t for t, c in counts.items() for _ in range(c)]
    rng.shuffle(tiers)

    split: dict[int, str] = {}
    for t in counts:
        idxs = [i for i, x in enumerate(tiers) if x == t]
        rng.shuffle(idxs)
        for i in idxs[: round(len(idxs) * args.heldout_frac)]:
            split[i] = "heldout"

    # Balanced trap schedule: within each (tier, split), trap_density of all invoice slots get a
    # trap, cycling evenly through trap types; the rest are clean. Shuffled, then dealt to tasks.
    trap_plan: dict[int, list[str]] = {}
    for t in counts:
        for sp in ("train", "heldout"):
            idxs = [i for i, x in enumerate(tiers) if x == t and split.get(i, "train") == sp]
            n_inv = TIERS[t].n_invoices
            slots = len(idxs) * n_inv
            n_trap = round(slots * TIERS[t].trap_density)
            start = rng.randrange(len(TRAPS))
            labels = [TRAPS[(start + k) % len(TRAPS)] for k in range(n_trap)] + ["clean"] * (slots - n_trap)
            rng.shuffle(labels)
            for j, i in enumerate(idxs):
                trap_plan[i] = labels[j * n_inv:(j + 1) * n_inv]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, tier in enumerate(tiers):
        tid = f"lg-{i:04d}"
        meta = generate_task(tid, task_seed(args.seed, i), TIERS[tier], out / tid, traps=trap_plan[i])
        rows.append({"task_id": tid, "tier": tier, "split": split.get(i, "train"), "path": tid,
                     "seed": meta["seed"], "n_invoices": len(meta["pending_invoice_ids"])})
        if (i + 1) % 50 == 0:
            print(f"  generated {i + 1}/{len(tiers)}")

    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "manifest_info.json").write_text(json.dumps(
        {"args": vars(args), "counts": counts, "python": platform.python_version()}, indent=2))

    c = Counter((r["tier"], r["split"]) for r in rows)
    print(f"\nwrote {len(rows)} tasks to {out}")
    for t in counts:
        print(f"  {t:<7} train={c[(t, 'train')]:>4}  heldout={c[(t, 'heldout')]:>4}")


if __name__ == "__main__":
    main()
