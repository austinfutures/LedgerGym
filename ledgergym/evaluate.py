"""Shared evaluation loop: run a policy over tasks, verify the final state, aggregate."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from .env import LedgerEnv
from .rewards import REWARD_DESIGNS, RewardConfig, compute_rewards
from .verifier import CORRECT, PAID_FRAUD, UNSAFE, verify

Policy = Callable  # (env, obs) -> final_text


def load_manifest(root: str | Path, split: str = "all", limit: int | None = None) -> list[dict]:
    root = Path(root)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    rows = rows[:limit] if limit else rows
    for r in rows:
        r["dir"] = root / r["path"]
    return rows


def run_episode(task_dir: str | Path, policy: Policy, max_tool_calls: int = 6,
                cfg: RewardConfig | None = None) -> dict:
    env = LedgerEnv(task_dir, max_tool_calls=max_tool_calls)
    try:
        obs = env.reset()
        final = policy(env, obs)
        v = verify(task_dir, env.finish(final or ""))
        rec = env.record()
    finally:
        env.close()
    return {"verdict": v, "rewards": compute_rewards(v, cfg), "record": rec}


def evaluate(tasks: list[dict], policy: Policy, max_tool_calls: int = 6,
             cfg: RewardConfig | None = None) -> dict:
    """tasks: dicts with 'dir', 'tier', 'task_id' (load_manifest provides these)."""
    agg = {"outcomes": Counter(), "by_trap": defaultdict(lambda: [0, 0]), "by_tier": defaultdict(lambda: [0, 0]),
           "rewards": defaultdict(list), "max_calls": 0, "tool_errors": 0, "crashed": [], "failures": [],
           "integrity": 0, "improper_dollars": 0.0, "n_tasks": 0}
    for t in tasks:
        try:
            ep = run_episode(t["dir"], policy, max_tool_calls, cfg)
        except Exception as e:
            agg["crashed"].append((t["task_id"], repr(e)[:300]))
            continue
        v, rw, rec = ep["verdict"], ep["rewards"], ep["record"]
        agg["n_tasks"] += 1
        for k in REWARD_DESIGNS:
            agg["rewards"][k].append(rw[k])
        agg["max_calls"] = max(agg["max_calls"], rec["tool_calls"])
        agg["tool_errors"] += sum(not s["ok"] for s in rec["trajectory"])
        agg["integrity"] += bool(v.integrity)
        agg["improper_dollars"] += v.improper_dollars
        for inv in v.invoices:
            ok = inv["outcome"] in CORRECT
            agg["outcomes"][inv["outcome"]] += 1
            for bucket in (agg["by_trap"][inv["trap"]], agg["by_tier"][t["tier"]]):
                bucket[0] += ok
                bucket[1] += 1
            if not ok:
                agg["failures"].append((t["task_id"], inv))
    return agg


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summary(agg: dict) -> dict:
    n_inv = sum(agg["outcomes"].values())
    return {
        "n_tasks": agg["n_tasks"], "n_invoices": n_inv,
        "accuracy": sum(agg["outcomes"][o] for o in CORRECT) / max(1, n_inv),
        **{k: _mean(agg["rewards"][k]) for k in REWARD_DESIGNS},
        "unsafe": sum(agg["outcomes"][o] for o in UNSAFE),
        "fraud_paid": agg["outcomes"][PAID_FRAUD],
        "improper_dollars": round(agg["improper_dollars"], 2),
        "tool_errors": agg["tool_errors"], "integrity_fails": agg["integrity"],
        "max_calls": agg["max_calls"], "crashed": len(agg["crashed"]),
        "outcomes": dict(agg["outcomes"]),
    }
