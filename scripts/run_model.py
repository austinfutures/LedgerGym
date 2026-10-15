"""Run a model on LedgerGym tasks; append one JSONL record per episode. Resumable.

    python scripts/run_model.py --model qwen1.5b --split heldout
    python scripts/run_model.py --model qwen1.5b --split heldout --prompt guided_sql --nudges 2
    python scripts/run_model.py --model qwen3b --split heldout --tiers easy,medium
    python scripts/run_model.py --model gemini-flash --split heldout --rpm 5 --rpd 100
    python scripts/run_model.py --model qwen1.5b --split train --samples 4 --temperature 0.8   # RFT data

Interrupt any time (Ctrl+C) and re-run the same command: finished episodes are skipped and
model responses come from the disk cache, so nothing is paid twice. If the daily quota runs
out, it stops cleanly; run the same command again after midnight Pacific.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from ledgergym.agent import run_and_grade
from ledgergym.evaluate import load_manifest
from ledgergym.llm import PRESETS, BackendError, QuotaExhausted, make_backend
from ledgergym.prompts import PROMPT_VARIANTS


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["episode_id"])
            except (json.JSONDecodeError, KeyError):
                pass          # a half-written last line from a hard kill is simply redone
    return done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(PRESETS))
    p.add_argument("--model-name", help="override the preset's model name")
    p.add_argument("--base-url", help="override the preset's endpoint")
    p.add_argument("--rpm", type=float, help="requests per minute limit")
    p.add_argument("--rpd", type=int, help="requests per day limit (ledger stops ~5%% below)")
    p.add_argument("--tasks", default="data/tasks")
    p.add_argument("--split", default="heldout", choices=["all", "train", "heldout"])
    p.add_argument("--tiers", default="easy,medium,hard")
    p.add_argument("--limit", type=int)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--prompt", default="base", choices=PROMPT_VARIANTS, help="system prompt variant")
    p.add_argument("--nudges", type=int, default=0,
                   help="times to tell the model to continue if it stops with invoices still pending")
    p.add_argument("--max-tool-calls", type=int, default=6)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-name")
    args = p.parse_args()

    if args.samples > 1 and args.temperature == 0:
        sys.exit("--samples > 1 needs --temperature > 0, otherwise every sample is identical.")

    tiers = set(args.tiers.split(","))
    tasks = [t for t in load_manifest(args.tasks, args.split) if t["tier"] in tiers]
    tasks = tasks[: args.limit] if args.limit else tasks

    harness = ("" if args.prompt == "base" else f"-{args.prompt}") + (f"-n{args.nudges}" if args.nudges else "")
    run_name = args.run_name or (f"{args.model}-{args.split}{harness}"
                                 + (f"-t{args.temperature}x{args.samples}" if args.samples > 1 else ""))
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "episodes.jsonl"
    cfg_path = run_dir / "run.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({**vars(args), "preset": PRESETS[args.model]}, indent=2))

    try:
        backend = make_backend(args.model, model=args.model_name, base_url=args.base_url, rpm=args.rpm,
                               rpd=args.rpd, temperature=args.temperature, max_tokens=args.max_tokens,
                               cache_dir=Path(args.runs_dir) / "cache")
    except BackendError as e:
        sys.exit(f"ERROR: {e}")

    done = load_done(out_path)
    todo = [(t, s) for t in tasks for s in range(args.samples) if f"{t['task_id']}#s{s}" not in done]
    print(f"run={run_name}  model={backend.model}  episodes: {len(done)} done, {len(todo)} to go")
    if backend.ledger.cap:
        print(f"quota: {backend.ledger.used_today()}/{backend.ledger.rpd} used today (stops at {backend.ledger.cap})")

    t_start, n = time.monotonic(), 0
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            for task, s in todo:
                rec = run_and_grade(task, backend, sample=s, max_tool_calls=args.max_tool_calls,
                                    prompt_variant=args.prompt, max_nudges=args.nudges)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                outs = ",".join(i["outcome"] for i in rec["verdict"]["invoices"])
                eta = (time.monotonic() - t_start) / n * (len(todo) - n) / 60
                print(f"[{len(done) + n}/{len(done) + len(todo)}] {rec['episode_id']:<12} {rec['tier']:<6} "
                      f"safety={rec['rewards']['partial_safety']:+.2f} calls={rec['tool_calls']} "
                      f"nudges={rec['nudges_used']} stop={rec['stop_reason']:<14} {rec['wall_s']:>6.1f}s  {outs}   "
                      f"(eta {eta:.0f} min)")
    except QuotaExhausted as e:
        print(f"\nSTOPPED: {e}\n{n} episodes saved this session. Re-run the same command to resume.")
        return
    except KeyboardInterrupt:
        print(f"\nInterrupted. {n} episodes saved this session. Re-run the same command to resume.")
        return
    except BackendError as e:
        sys.exit(f"\nERROR: {e}\n{n} episodes saved this session.")
    print(f"\nfinished: {n} new episodes  ({backend.stats['requests']} API requests, "
          f"{backend.stats['cache_hits']} cache hits)  ->  {out_path}")
    print(f"summarize: python scripts/summarize_runs.py {run_dir}")


if __name__ == "__main__":
    main()
