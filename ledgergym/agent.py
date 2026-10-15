"""Agent loop: a chat model drives LedgerEnv through tool calls, then the verifier grades it.

The loop is backend-agnostic. Anything with chat(messages, tools, sample) -> AssistantTurn
works: the OpenAI-compatible backend, the transformers backend (piece 9), or a scripted
fake in the tests.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .env import LedgerEnv
from .rewards import compute_rewards
from .verifier import verify


def run_agent(env: LedgerEnv, backend, sample: int = 0, max_turns: int | None = None,
              max_nudges: int = 0) -> dict:
    """max_nudges: if the model answers in text while invoices are still pending, reply with a
    user message listing what is left (nudge_prompt) instead of ending, up to this many times.
    Nudges do not use the tool budget."""
    from .prompts import nudge_prompt

    obs = env.reset()
    messages: list[dict] = [{"role": "system", "content": obs["system"]},
                            {"role": "user", "content": obs["user"]}]
    turns: list[dict] = []
    final_text, stop_reason, nudges = "", "max_turns", 0
    max_turns = max_turns or env.max_tool_calls + 3 + max_nudges

    for t in range(max_turns):
        if env.done:
            stop_reason = "budget"
            break
        turn = backend.chat(messages, obs["tools"], sample=sample)
        turns.append({"parse_mode": turn.parse_mode, "n_calls": len(turn.tool_calls), "usage": turn.usage,
                      "cached": turn.cached, "latency_s": round(turn.latency_s, 3),
                      "finish_reason": turn.finish_reason})
        if not turn.tool_calls:
            text = turn.content or ""
            messages.append({"role": "assistant", "content": text})
            pending = env.pending_ids()
            if pending and nudges < max_nudges:
                nudges += 1
                messages.append({"role": "user",
                                 "content": nudge_prompt(pending, env.max_tool_calls - env.calls)})
                continue
            final_text = text
            stop_reason = "final_answer" if text.strip() else "empty_response"
            break

        calls = [{"id": c.get("id") or f"call_{t}_{i}", "name": c["name"], "arguments": c["arguments"]}
                 for i, c in enumerate(turn.tool_calls)]
        messages.append({"role": "assistant", "content": turn.content or None,
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for c in calls]})
        for c in calls:
            try:
                args = json.loads(c["arguments"]) if c["arguments"] else {}
            except json.JSONDecodeError:
                args = c["arguments"]          # tools report "arguments must be a JSON object"
            res = env.step(c["name"], args)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": res.text})
    else:
        stop_reason = "budget" if env.done else "max_turns"

    env.finish(final_text)
    return {"messages": messages, "turns": turns, "final_text": final_text, "stop_reason": stop_reason,
            "nudges_used": nudges}


def _final_actions(db_path: Path) -> dict:
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        pays = conn.execute("SELECT payment_id, invoice_id, vendor_id, vendor_invoice_number, amount "
                            "FROM payments WHERE payment_id LIKE 'PMT-E%' ORDER BY payment_id").fetchall()
        flags = conn.execute("SELECT invoice_id, reason_code, note FROM exceptions ORDER BY exception_id").fetchall()
        mails = conn.execute("SELECT to_address, subject FROM outbox ORDER BY email_id").fetchall()
    finally:
        conn.close()
    return {"payments": [list(r) for r in pays], "exceptions": [list(r) for r in flags],
            "emails": [list(r) for r in mails]}


def run_and_grade(task: dict, backend, sample: int = 0, max_tool_calls: int = 6,
                  prompt_variant: str = "base", max_nudges: int = 0) -> dict:
    """task: manifest row with 'dir', 'task_id', 'tier', 'split'. Returns one JSONL-ready record."""
    env = LedgerEnv(task["dir"], max_tool_calls=max_tool_calls, prompt_variant=prompt_variant)
    t0 = time.monotonic()
    try:
        out = run_agent(env, backend, sample=sample, max_nudges=max_nudges)
        v = verify(task["dir"], env.db_path)
        actions = _final_actions(env.db_path)
        rec = env.record()
    finally:
        env.close()
    return {
        "episode_id": f"{task['task_id']}#s{sample}", "task_id": task["task_id"], "tier": task["tier"],
        "split": task.get("split"), "sample": sample,
        "backend": getattr(backend, "name", "?"), "model": getattr(backend, "model", "?"),
        "harness": {"prompt": prompt_variant, "max_nudges": max_nudges}, "nudges_used": out["nudges_used"],
        "stop_reason": out["stop_reason"], "tool_calls": rec["tool_calls"], "max_tool_calls": max_tool_calls,
        "trajectory": rec["trajectory"], "messages": out["messages"], "turns": out["turns"],
        "final_text": out["final_text"], "final_actions": actions,
        "verdict": v.to_dict(), "rewards": compute_rewards(v),
        "wall_s": round(time.monotonic() - t0, 2),
    }
