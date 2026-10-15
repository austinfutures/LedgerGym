"""Episode environment.

Each reset() copies the task's erp.sqlite into a private run directory, so the generated
task is never modified and episodes can be replayed. The environment never opens
ground_truth.json; only the verifier does.

    env = LedgerEnv("data/tasks/lg-0000", max_tool_calls=6)
    obs = env.reset()                  # obs["system"], obs["user"], obs["tools"], obs["pending"]
    r = env.step("read_document", {"doc_ids": ["DOC-0002"]})
    ...
    db_path = env.finish("summary text")   # final DB state, ready for the verifier
    env.close()                        # deletes the run dir if it was temporary
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .prompts import system_prompt, task_prompt
from .tools import TOOL_SPECS, ERPTools, ToolResult


class LedgerEnv:
    def __init__(self, task_dir: str | Path, max_tool_calls: int = 6, run_dir: str | Path | None = None,
                 prompt_variant: str = "base"):
        self.task_dir = Path(task_dir)
        self.prompt_variant = prompt_variant
        self.meta = json.loads((self.task_dir / "task.json").read_text())
        self.max_tool_calls = max_tool_calls
        self._run_dir_arg = run_dir
        self.run_dir: Path | None = None
        self._own_run_dir = False
        self.tools: ERPTools | None = None
        self.calls = 0
        self.done = True
        self.trajectory: list[dict] = []
        self.final_text: str | None = None

    @property
    def db_path(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("no active episode; call reset()")
        return self.run_dir / "erp.sqlite"

    def reset(self) -> dict:
        self.close()
        if self._run_dir_arg is not None:
            self.run_dir = Path(self._run_dir_arg)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._own_run_dir = False
        else:
            self.run_dir = Path(tempfile.mkdtemp(prefix="ledgergym-"))
            self._own_run_dir = True
        shutil.copyfile(self.task_dir / "erp.sqlite", self.db_path)
        self.tools = ERPTools(self.db_path, self.task_dir / "inbox", self.meta["base_date"])

        pending = [{"invoice_id": a, "doc_id": b} for a, b in self.tools.conn.execute(
            "SELECT invoice_id, doc_id FROM invoices WHERE status='pending' ORDER BY invoice_id")]
        self.calls, self.done, self.trajectory, self.final_text = 0, False, [], None
        return {"task_id": self.meta["task_id"],
                "system": system_prompt(self.max_tool_calls, self.prompt_variant),
                "user": task_prompt(self.meta["base_date"], pending),
                "tools": TOOL_SPECS,
                "pending": pending}

    def step(self, name: str, args: dict[str, Any] | None = None) -> ToolResult:
        if self.done or self.tools is None:
            return ToolResult("ERROR: the episode is over; no more tool calls are accepted.", ok=False)
        args = {} if args is None else args
        self.calls += 1
        res = self.tools.call(name, args)
        if self.calls >= self.max_tool_calls:
            self.done = True
            res.text += f"\n[Tool budget of {self.max_tool_calls} calls is used up. The episode is over.]"
        self.trajectory.append({"step": self.calls, "tool": name, "args": args,
                                "ok": res.ok, "mutated": res.mutated, "result": res.text})
        return res

    def pending_ids(self) -> list[str]:
        """Invoices still pending in this episode (used by the harness to nudge, never by the verifier)."""
        if self.tools is None:
            return []
        return [r[0] for r in self.tools.conn.execute(
            "SELECT invoice_id FROM invoices WHERE status='pending' ORDER BY invoice_id")]

    def finish(self, final_text: str = "") -> Path:
        """End the episode (agent stopped or budget ran out). Returns the final DB path."""
        self.done = True
        self.final_text = final_text
        if self.tools is not None:
            self.tools.close()
            self.tools = None
        return self.db_path

    def record(self) -> dict:
        return {"task_id": self.meta["task_id"], "tier": self.meta["difficulty"]["name"],
                "max_tool_calls": self.max_tool_calls, "tool_calls": self.calls,
                "trajectory": self.trajectory, "final_text": self.final_text}

    def close(self) -> None:
        if self.tools is not None:
            self.tools.close()
            self.tools = None
        if self.run_dir is not None and self._own_run_dir:
            shutil.rmtree(self.run_dir, ignore_errors=True)
        self.run_dir = None
