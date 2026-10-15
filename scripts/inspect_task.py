"""Pretty-print one task: metadata, table sizes, inbox docs, and (optionally) ground truth.

    python scripts/inspect_task.py data/tasks/lg-0007
    python scripts/inspect_task.py data/tasks/lg-0007 --no-gt
"""
import json
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    d = Path(sys.argv[1])
    show_gt = "--no-gt" not in sys.argv

    print("=== task.json ===")
    print((d / "task.json").read_text())

    conn = sqlite3.connect(d / "erp.sqlite")
    print("=== table sizes ===")
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        print(f"  {t:<22} {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")

    print("\n=== bank change log ===")
    for row in conn.execute("SELECT * FROM vendor_bank_changes"):
        print(" ", row)

    print("\n=== inbox ===")
    for doc_id, kind, fname, received, sender, subject in conn.execute("SELECT * FROM inbox ORDER BY doc_id"):
        print(f"\n--- {doc_id} [{kind}] from {sender} on {received}: {subject}")
        print((d / "inbox" / fname).read_text())
    conn.close()

    if show_gt:
        print("=== ground truth ===")
        print((d / "ground_truth.json").read_text())


if __name__ == "__main__":
    main()
