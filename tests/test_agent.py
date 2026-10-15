"""Agent loop, text fallback parsing, cache and quota tests. No model or network needed:
a scripted fake backend plays the model, and the API layer is stubbed."""
import json

import pytest

from ledgergym.agent import run_and_grade
from ledgergym.generator import generate_task
from ledgergym.llm import (AssistantTurn, ChatBackend, QuotaExhausted, QuotaLedger, RateLimited,
                           extract_text_tool_calls, turn_from_raw)


class FakeBackend:
    """Plays back a fixed list of turns. Each turn: list of (name, args) tool calls, or a str final."""
    name, model = "fake", "fake"

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools, sample=0):
        step = self.script.pop(0) if self.script else "done"
        if isinstance(step, str):
            return AssistantTurn(step, [], "none")
        calls = [{"id": "", "name": n, "arguments": a if isinstance(a, str) else json.dumps(a)} for n, a in step]
        return AssistantTurn(None, calls, "native")


@pytest.fixture
def task(tmp_path):
    d = tmp_path / "t"
    generate_task("t", 3, "medium", d, traps=["clean", "bank_change_fraud"])
    gt = json.loads((d / "ground_truth.json").read_text())["invoices"]
    meta = json.loads((d / "task.json").read_text())
    return {"dir": d, "task_id": "t", "tier": "medium", "split": "heldout"}, gt, meta


def test_scripted_perfect_episode(task):
    t, gt, _ = task
    clean, fraud = gt
    script = [
        [("read_document", {"doc_ids": [clean["doc_id"], fraud["doc_id"]]})],
        [("query_db", {"sql": "SELECT * FROM vendor_bank_changes"})],
        [("post_payment", {"invoice_id": clean["invoice_id"], "vendor_id": clean["vendor_id"],
                           "amount": clean["amount"]}),
         ("flag_exception", {"invoice_id": fraud["invoice_id"], "reason_code": "bank_change_fraud"})],
        "Paid AP-0001, flagged AP-0002 as fraud.",
    ]
    rec = run_and_grade(t, FakeBackend(script))
    assert rec["rewards"] == {"binary": 1.0, "partial": 1.0, "partial_safety": 1.0}
    assert rec["stop_reason"] == "final_answer" and rec["tool_calls"] == 4
    assert len(rec["final_actions"]["payments"]) == 1 and len(rec["final_actions"]["exceptions"]) == 1
    # the message history is valid OpenAI format: each tool result answers a tool_call id
    ids = [tc["id"] for m in rec["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    assert rec["messages"][-1] == {"role": "assistant", "content": "Paid AP-0001, flagged AP-0002 as fraud."}
    assert ids == [m["tool_call_id"] for m in rec["messages"] if m["role"] == "tool"]


def test_budget_stops_a_looping_model(task):
    t, _, _ = task
    rec = run_and_grade(t, FakeBackend([[("query_db", {"sql": "SELECT 1"})]] * 20))
    assert rec["stop_reason"] == "budget" and rec["tool_calls"] == 6


def test_bad_arguments_cost_a_call_and_are_logged(task):
    t, _, _ = task
    rec = run_and_grade(t, FakeBackend([[("post_payment", "{not json")], [("pay_invoice", {})], "done"]))
    assert [s["ok"] for s in rec["trajectory"]] == [False, False]
    assert "JSON object" in rec["trajectory"][0]["result"] and "unknown tool" in rec["trajectory"][1]["result"]
    assert rec["rewards"]["partial_safety"] == 0.0


def test_empty_response_stops(task):
    t, _, _ = task
    rec = run_and_grade(t, FakeBackend([""]))
    assert rec["stop_reason"] == "empty_response" and rec["tool_calls"] == 0


@pytest.mark.parametrize("content,name", [
    ('<tool_call>\n{"name": "query_db", "arguments": {"sql": "SELECT 1"}}\n</tool_call>', "query_db"),
    ('Let me check.\n```json\n{"name": "read_document", "arguments": {"doc_ids": ["DOC-0001"]}}\n```', "read_document"),
    ('{"name": "flag_exception", "parameters": {"invoice_id": "AP-0001", "reason_code": "duplicate"}}', "flag_exception"),
])
def test_text_tool_call_fallback(content, name):
    calls = extract_text_tool_calls(content)
    assert len(calls) == 1 and calls[0]["name"] == name
    assert isinstance(json.loads(calls[0]["arguments"]), dict)


def test_no_false_positive_tool_calls():
    assert extract_text_tool_calls("I paid AP-0001 and flagged AP-0002 as a duplicate.") == []
    assert extract_text_tool_calls('{"summary": "done"}') == []


def test_turn_from_raw_prefers_native_calls():
    raw = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "x", "function": {"name": "query_db", "arguments": '{"sql": "SELECT 1"}'}}]},
        "finish_reason": "tool_calls"}]}
    turn = turn_from_raw(raw)
    assert turn.parse_mode == "native" and turn.tool_calls[0]["name"] == "query_db"


class StubBackend(ChatBackend):
    """ChatBackend with the network call replaced."""
    def __init__(self, tmp_path, responder, rpd=None):
        super().__init__("stub", "http://stub", "stub-model", rpd=rpd, cache_dir=tmp_path / "cache")
        self.responder, self.calls = responder, 0

    def _request(self, messages, tools):
        self.calls += 1
        return self.responder()


FINAL = lambda: {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def test_cache_replays_without_spending_quota(tmp_path):
    b = StubBackend(tmp_path, FINAL, rpd=100)
    msgs = [{"role": "user", "content": "hi"}]
    first, second = b.chat(msgs, []), b.chat(msgs, [])
    assert b.calls == 1 and not first.cached and second.cached
    assert b.ledger.used_today() == 1
    b.chat(msgs, [], sample=1)            # a different sample index is a different request
    assert b.calls == 2


def test_server_daily_quota_raises(tmp_path):
    def daily():
        raise RateLimited("429 Quota exceeded for metric generate_content_free_tier_requests, limit per day")
    with pytest.raises(QuotaExhausted):
        StubBackend(tmp_path, daily).chat([{"role": "user", "content": "x"}], [])


def test_ledger_stops_before_the_cap(tmp_path):
    ledger = QuotaLedger(tmp_path / "q.json", rpd=20, margin=0.1)   # stops at 18
    for _ in range(17):
        ledger.check()
        ledger.add()
    ledger.check()
    ledger.add()
    with pytest.raises(QuotaExhausted):
        ledger.check()


def test_nudge_rescues_a_model_that_narrates(task):
    t, gt, _ = task
    clean, fraud = gt
    script = [
        "First I will read the invoice documents, then query the database.",   # narrates, no call
        [("read_document", {"doc_ids": [clean["doc_id"], fraud["doc_id"]]})],
        [("post_payment", {"invoice_id": clean["invoice_id"], "vendor_id": clean["vendor_id"],
                           "amount": clean["amount"]}),
         ("flag_exception", {"invoice_id": fraud["invoice_id"], "reason_code": "bank_change_fraud"})],
        "Done.",
    ]
    rec = run_and_grade(t, FakeBackend(script), max_nudges=2)
    assert rec["nudges_used"] == 1 and rec["stop_reason"] == "final_answer"
    assert rec["rewards"]["binary"] == 1.0
    nudge = [m for m in rec["messages"] if m["role"] == "user"][1]["content"]
    assert "AP-0001" in nudge and "AP-0002" in nudge and "6 tool call(s) left" in nudge
    assert rec["harness"] == {"prompt": "base", "max_nudges": 2}


def test_no_nudge_when_disabled_or_nothing_pending(task):
    t, _, _ = task
    rec = run_and_grade(t, FakeBackend(["I will now read the documents."]), max_nudges=0)
    assert rec["nudges_used"] == 0 and rec["stop_reason"] == "final_answer"


def test_nudges_are_capped(task):
    t, _, _ = task
    rec = run_and_grade(t, FakeBackend(["thinking..."] * 10), max_nudges=2)
    assert rec["nudges_used"] == 2 and rec["stop_reason"] == "final_answer" and rec["tool_calls"] == 0


@pytest.mark.parametrize("variant,must,must_not", [
    ("base", "Pending invoices have NULL", "How to work"),
    ("guided", "How to work", "Example of step 2"),
    ("guided_sql", "Example of step 2", None),
])
def test_prompt_variants(task, variant, must, must_not):
    from ledgergym.env import LedgerEnv
    env = LedgerEnv(task[0]["dir"], prompt_variant=variant)
    system = env.reset()["system"]
    env.close()
    assert must in system and (must_not is None or must_not not in system)
