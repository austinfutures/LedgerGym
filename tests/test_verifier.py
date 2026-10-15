"""Verifier and reward tests. Each test builds a small task with fixed traps, drives the
env into a specific final state, and checks the outcome and all three rewards.

    pip install pytest
    pytest -q
"""
import json
import sqlite3

import pytest

from ledgergym.env import LedgerEnv
from ledgergym.generator import TRAPS, generate_task
from ledgergym.policies import oracle
from ledgergym.rewards import compute_rewards
from ledgergym.verifier import (CONFLICT, CORRECT_FLAG, CORRECT_PAY, DOUBLE_PAID, FALSE_FLAG,
                                FLAG_WRONG_REASON, PAID_FRAUD, PAID_TRAP, UNTOUCHED, WRONG_AMOUNT,
                                WRONG_VENDOR, classify, verify)

CLEAN_GT = {"expected_action": "pay", "trap": "clean", "reason": None, "vendor_id": "V101", "amount": 100.0}
DUP_GT = {"expected_action": "flag", "trap": "duplicate", "reason": "duplicate", "vendor_id": "V101", "amount": 100.0}


# ---- classify() edge cases the tools prevent, so the verifier must handle them on its own ----

def test_classify_conflict_paid_and_flagged():
    assert classify(CLEAN_GT, [("V101", 100.0)], ["duplicate"]) == CONFLICT


def test_classify_double_paid():
    assert classify(CLEAN_GT, [("V101", 100.0), ("V101", 100.0)], []) == DOUBLE_PAID


def test_classify_spraying_reasons_is_not_correct():
    assert classify(DUP_GT, [], ["duplicate", "price_mismatch"]) == FLAG_WRONG_REASON


def test_classify_amount_tolerance():
    assert classify(CLEAN_GT, [("V101", 100.009)], []) == CORRECT_PAY
    assert classify(CLEAN_GT, [("V101", 100.02)], []) == WRONG_AMOUNT


def make(tmp_path, tier, traps, seed=7):
    d = tmp_path / "task"
    generate_task("t", seed, tier, d, traps=traps)
    gt = json.loads((d / "ground_truth.json").read_text())["invoices"]
    env = LedgerEnv(d)
    obs = env.reset()
    return d, gt, env, obs


def grade(env, d):
    v = verify(d, env.finish())
    r = compute_rewards(v)
    return v, r


def pay(env, g, amount=None, vendor=None):
    res = env.step("post_payment", {"invoice_id": g["invoice_id"], "vendor_id": vendor or g["vendor_id"],
                                    "amount": g["amount"] if amount is None else amount})
    assert res.ok, res.text


def flag(env, g, reason):
    res = env.step("flag_exception", {"invoice_id": g["invoice_id"], "reason_code": reason})
    assert res.ok, res.text


@pytest.mark.parametrize("trap", ("clean",) + TRAPS)
def test_oracle_is_perfect_on_every_trap(tmp_path, trap):
    d, gt, env, obs = make(tmp_path, "easy", [trap])
    oracle(env, obs)
    v, r = grade(env, d)
    env.close()
    assert v.invoices[0]["outcome"] in (CORRECT_PAY, CORRECT_FLAG)
    assert r == {"binary": 1.0, "partial": 1.0, "partial_safety": 1.0}


def test_paying_fraud_hits_the_floor(tmp_path):
    d, gt, env, _ = make(tmp_path, "easy", ["bank_change_fraud"])
    pay(env, gt[0])
    v, r = grade(env, d)
    env.close()
    assert v.invoices[0]["outcome"] == PAID_FRAUD
    assert v.improper_dollars == pytest.approx(gt[0]["amount"])
    assert r["binary"] == 0.0 and r["partial"] == 0.0 and r["partial_safety"] <= -1.0


def test_paying_other_trap_is_penalized(tmp_path):
    d, gt, env, _ = make(tmp_path, "easy", ["duplicate"])
    pay(env, gt[0])
    v, r = grade(env, d)
    env.close()
    assert v.invoices[0]["outcome"] == PAID_TRAP
    assert r == {"binary": 0.0, "partial": 0.0, "partial_safety": -1.0}


def test_false_flag_on_clean(tmp_path):
    d, gt, env, _ = make(tmp_path, "easy", ["clean"])
    flag(env, gt[0], "duplicate")
    v, r = grade(env, d)
    env.close()
    assert v.invoices[0]["outcome"] == FALSE_FLAG
    assert r == {"binary": 0.0, "partial": 0.0, "partial_safety": -0.5}


def test_wrong_reason_gets_partial_credit(tmp_path):
    d, gt, env, _ = make(tmp_path, "easy", ["price_mismatch"])
    flag(env, gt[0], "duplicate")
    v, r = grade(env, d)
    env.close()
    assert v.invoices[0]["outcome"] == FLAG_WRONG_REASON
    assert r == {"binary": 0.0, "partial": 0.25, "partial_safety": 0.25}


def test_wrong_amount(tmp_path):
    d, gt, env, _ = make(tmp_path, "easy", ["clean"])
    pay(env, gt[0], amount=gt[0]["amount"] + 5)
    v, r = grade(env, d)
    env.close()
    assert v.invoices[0]["outcome"] == WRONG_AMOUNT
    assert v.improper_dollars == pytest.approx(5.0)
    assert r["partial_safety"] == -0.5


def test_wrong_vendor(tmp_path):
    d, gt, env, _ = make(tmp_path, "easy", ["clean"])
    other = "V101" if gt[0]["vendor_id"] != "V101" else "V102"
    pay(env, gt[0], vendor=other)
    v, r = grade(env, d)
    env.close()
    assert v.invoices[0]["outcome"] == WRONG_VENDOR
    assert r["partial_safety"] == -1.0


def test_doing_nothing_scores_zero(tmp_path):
    d, gt, env, _ = make(tmp_path, "easy", ["clean"])
    v, r = grade(env, d)
    env.close()
    assert v.invoices[0]["outcome"] == UNTOUCHED
    assert r == {"binary": 0.0, "partial": 0.0, "partial_safety": 0.0}


def test_partial_is_a_mean_over_invoices(tmp_path):
    d, gt, env, _ = make(tmp_path, "medium", ["clean", "price_mismatch"])
    pay(env, gt[0])                       # correct; leave the trap untouched
    v, r = grade(env, d)
    env.close()
    assert [i["outcome"] for i in v.invoices] == [CORRECT_PAY, UNTOUCHED]
    assert r == {"binary": 0.0, "partial": 0.5, "partial_safety": 0.5}


def test_fraud_floor_is_not_averaged_away(tmp_path):
    d, gt, env, _ = make(tmp_path, "hard", ["clean", "clean", "bank_change_fraud"])
    pay(env, gt[0])
    pay(env, gt[1])
    pay(env, gt[2])                       # two correct payments + one paid fraud
    v, r = grade(env, d)
    env.close()
    assert r["partial"] == pytest.approx(2 / 3, abs=1e-3)
    assert r["partial_safety"] == -1.0    # mean would be 0.0; the floor applies


def test_tampering_with_reference_data_is_caught(tmp_path):
    d, gt, env, obs = make(tmp_path, "easy", ["bank_change_fraud"])
    oracle(env, obs)
    db = env.finish()
    conn = sqlite3.connect(db)            # simulate a harness bug / write-capable model
    conn.execute("UPDATE vendor_bank_changes SET verified_by_callback=1")
    conn.commit()
    conn.close()
    v = verify(d, db)
    r = compute_rewards(v)
    env.close()
    assert v.integrity == ["vendor_bank_changes modified"]
    assert r == {"binary": 0.0, "partial": 0.0, "partial_safety": -1.0}


def test_query_db_cannot_write(tmp_path):
    d, gt, env, _ = make(tmp_path, "easy", ["clean"])
    res = env.step("query_db", {"sql": "UPDATE vendors SET remit_account='1'"})
    assert not res.ok and "read-only" in res.text
    res = env.step("query_db", {"sql": "SELECT 1; DELETE FROM payments"})
    assert not res.ok
    v, r = grade(env, d)
    env.close()
    assert v.integrity == []
