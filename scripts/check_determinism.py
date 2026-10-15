"""Generate tasks twice with the same seed and confirm they are byte-for-byte identical
(DB contents via iterdump, inbox docs, json files). Also confirms different seeds differ.

    python scripts/check_determinism.py
"""
import hashlib
import sqlite3
import tempfile
from pathlib import Path

from ledgergym.generator import TIERS, generate_task


def fingerprint(d: Path) -> str:
    h = hashlib.sha256()
    conn = sqlite3.connect(d / "erp.sqlite")
    for line in conn.iterdump():
        h.update(line.encode())
    conn.close()
    for f in sorted((d / "inbox").iterdir()):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    for name in ("task.json", "ground_truth.json"):
        h.update((d / name).read_bytes())
    return h.hexdigest()


def main() -> None:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for tier in TIERS:
            prints = []
            for seed in range(5):
                da, db = tmp / f"{tier}-{seed}-a", tmp / f"{tier}-{seed}-b"
                generate_task("t", seed, tier, da)
                generate_task("t", seed, tier, db)
                a, b = fingerprint(da), fingerprint(db)
                if a != b:
                    failures += 1
                    print(f"FAIL  {tier} seed={seed}: same seed produced different tasks")
                prints.append(a)
            distinct = len(set(prints))
            print(f"{tier:<7} same-seed reproducible: {'yes' if failures == 0 else 'NO'}   "
                  f"distinct across 5 seeds: {distinct}/5")
            if distinct != 5:
                failures += 1
    print("\nOK" if failures == 0 else f"\n{failures} problem(s)")


if __name__ == "__main__":
    main()
