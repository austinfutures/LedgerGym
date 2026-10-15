"""Print one episode from a run as a readable transcript.

    python scripts/show_episode.py runs/qwen1.5b-heldout                 # first episode
    python scripts/show_episode.py runs/qwen1.5b-heldout --index 2
    python scripts/show_episode.py runs/qwen1.5b-heldout --id lg-0010#s0
    python scripts/show_episode.py runs/qwen1.5b-heldout --system --width 0   # full text, incl. system prompt
"""
import argparse
import json
from pathlib import Path


def clip(text, width: int) -> str:
    text = "" if text is None else str(text)
    return text if width <= 0 or len(text) <= width else text[:width] + f" ... [+{len(text) - width} chars]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--id")
    p.add_argument("--width", type=int, default=1200, help="max chars per message (0 = no limit)")
    p.add_argument("--system", action="store_true", help="also print the system prompt")
    args = p.parse_args()

    path = Path(args.run)
    path = path / "episodes.jsonl" if path.is_dir() else path
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rec = next((r for r in recs if r["episode_id"] == args.id), None) if args.id else recs[args.index]
    if rec is None:
        raise SystemExit(f"no episode {args.id}")

    print(f"=== {rec['episode_id']}  tier={rec['tier']}  model={rec['model']}  stop={rec['stop_reason']}  "
          f"calls={rec['tool_calls']}/{rec['max_tool_calls']}")
    for i, m in enumerate(rec["messages"]):
        role = m["role"]
        if role == "system" and not args.system:
            print(f"\n[{i}] SYSTEM  ({len(m['content'])} chars, use --system to show)")
            continue
        print(f"\n[{i}] {role.upper()}")
        if m.get("content"):
            print(clip(m["content"], args.width))
        for tc in m.get("tool_calls") or []:
            print(f"  -> CALL {tc['function']['name']}({clip(tc['function']['arguments'], args.width)})")
    last = rec["messages"][-1] if rec["messages"] else {}
    already_shown = last.get("role") == "assistant" and not last.get("tool_calls")
    if rec.get("final_text") and not already_shown:      # episodes logged before the agent.py fix
        print(f"\n[final] ASSISTANT (final answer)\n{clip(rec['final_text'], args.width)}")
    print("\n--- turns:")
    for t in rec["turns"]:
        u = t.get("usage") or {}
        print(f"  parse={t['parse_mode']:<13} calls={t['n_calls']} finish={t.get('finish_reason')} "
              f"prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} "
              f"{t['latency_s']}s")
    print("--- verdict:")
    for inv in rec["verdict"]["invoices"]:
        print(f"  {inv['invoice_id']}: trap={inv['trap']:<18} -> {inv['outcome']}")
    print(f"--- rewards: {rec['rewards']}")


if __name__ == "__main__":
    main()
