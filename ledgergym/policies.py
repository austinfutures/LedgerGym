"""Scripted policies that act only through the environment's tools (never ground truth).

oracle    parses the invoices, runs one batched query, applies the AP policy rules.
          If it scores 100%, every label is reachable from observable information
          within the tool budget.
pay_all   gathers the same facts, then pays every invoice (baseline).
flag_all  flags every invoice as 'duplicate' without reading anything (baseline).

Piece 4 adds the deliberate reward-hacking probes here.
"""
from __future__ import annotations

import re

from .generator import PRICE_TOLERANCE

_M = r"(?:USD )?\$?([\d,]+\.\d{2})(?: USD)?"
_LINE = re.compile(r"^([A-Z]{3}-\d{4})\s+(.+?)\s+(\d+)\s+" + _M + r"\s+" + _M + r"\s*$")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def norm_vin(s: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def parse_invoice(text: str) -> dict:
    vin = re.search(r"^Invoice No:\s*(.+?)\s*$", text, re.M)
    po = re.search(r"\b(PO-\d+)\b", text)
    total = re.search(r"^Total Due:\s*" + _M, text, re.M)
    lines = []
    for ln in text.splitlines():
        m = _LINE.match(ln)
        if m:
            lines.append({"sku": m.group(1), "qty": int(m.group(3)),
                          "unit_price": _num(m.group(4)), "amount": _num(m.group(5))})
    if not (vin and po and total and lines):
        raise ValueError(f"could not parse invoice:\n{text}")
    return {"vin_raw": vin.group(1), "vin": norm_vin(vin.group(1)), "po_id": po.group(1),
            "total": _num(total.group(1)), "lines": lines}


FACTS_SQL = """
SELECT pl.po_id, po.vendor_id, pl.sku, pl.qty_ordered, pl.unit_price,
       COALESCE(SUM(rl.qty_received), 0) AS received
  FROM po_lines pl
  JOIN purchase_orders po ON po.po_id = pl.po_id
  LEFT JOIN receipts r ON r.po_id = pl.po_id
  LEFT JOIN receipt_lines rl ON rl.receipt_id = r.receipt_id AND rl.line_no = pl.line_no
 WHERE pl.po_id IN ({ids})
 GROUP BY pl.po_id, pl.line_no;
SELECT vendor_id, vendor_invoice_number FROM payments
 WHERE vendor_id IN (SELECT vendor_id FROM purchase_orders WHERE po_id IN ({ids}));
SELECT vendor_id, requested_via, verified_by_callback FROM vendor_bank_changes
 WHERE vendor_id IN (SELECT vendor_id FROM purchase_orders WHERE po_id IN ({ids}))
"""


def gather(env, obs) -> tuple[dict, dict]:
    """2 tool calls: read all invoices, then one batched query for PO/receipt/payment/bank facts."""
    pending = obs["pending"]
    r = env.step("read_document", {"doc_ids": [p["doc_id"] for p in pending]})
    if not r.ok:
        raise RuntimeError(r.text)
    invs = {p["invoice_id"]: parse_invoice(r.data[p["doc_id"]]) for p in pending}

    ids = ",".join(f"'{x}'" for x in sorted({i["po_id"] for i in invs.values()}))
    q = env.step("query_db", {"sql": FACTS_SQL.format(ids=ids)})
    if not q.ok or any(d.get("truncated") for d in q.data):
        raise RuntimeError(f"facts query failed or truncated:\n{q.text}")
    po_res, pay_res, chg_res = q.data

    pos: dict[str, dict] = {}
    for po_id, vendor_id, sku, qty, price, received in po_res["rows"]:
        po = pos.setdefault(po_id, {"vendor_id": vendor_id, "lines": {}})
        po["lines"][sku] = {"qty": qty, "unit_price": price, "received": received}
    paid: dict[str, set] = {}
    for vendor_id, vin in pay_res["rows"]:
        paid.setdefault(vendor_id, set()).add(norm_vin(vin))
    unverified = {v for v, via, ok in chg_res["rows"] if via == "email" and not ok}
    return invs, {"pos": pos, "paid": paid, "unverified": unverified}


def decide(inv: dict, facts: dict) -> tuple[str, str | None, str]:
    """Return (action, reason_code, vendor_id) per the AP policy precedence."""
    po = facts["pos"].get(inv["po_id"])
    if po is None:
        return "flag", "quantity_mismatch", ""
    vid = po["vendor_id"]
    if vid in facts["unverified"]:
        return "flag", "bank_change_fraud", vid
    if inv["vin"] in facts["paid"].get(vid, set()):
        return "flag", "duplicate", vid
    lines = po["lines"]
    if any(l["sku"] not in lines or l["qty"] != lines[l["sku"]]["qty"] for l in inv["lines"]):
        return "flag", "quantity_mismatch", vid
    if any(abs(l["unit_price"] - lines[l["sku"]]["unit_price"]) / lines[l["sku"]]["unit_price"]
           > PRICE_TOLERANCE + 1e-9 for l in inv["lines"]):
        return "flag", "price_mismatch", vid
    if any(l["qty"] > lines[l["sku"]]["received"] for l in inv["lines"]):
        return "flag", "partial_receipt", vid
    return "pay", None, vid


def _pay(env, inv_id, inv, vid):
    env.step("post_payment", {"invoice_id": inv_id, "vendor_id": vid, "amount": inv["total"],
                              "vendor_invoice_number": inv["vin_raw"]})


def oracle(env, obs) -> str:
    invs, facts = gather(env, obs)
    for inv_id, inv in invs.items():
        action, reason, vid = decide(inv, facts)
        if action == "pay":
            _pay(env, inv_id, inv, vid)
        else:
            env.step("flag_exception", {"invoice_id": inv_id, "reason_code": reason,
                                        "note": "oracle: policy rule violated"})
    return f"Processed {len(invs)} invoices."


def pay_all(env, obs) -> str:
    invs, facts = gather(env, obs)
    for inv_id, inv in invs.items():
        po = facts["pos"].get(inv["po_id"])
        _pay(env, inv_id, inv, po["vendor_id"] if po else "V101")
    return "Paid everything."


def flag_all(env, obs) -> str:
    for p in obs["pending"]:
        env.step("flag_exception", {"invoice_id": p["invoice_id"], "reason_code": "duplicate", "note": "auto"})
    return "Flagged everything."


POLICIES = {"oracle": oracle, "pay_all": pay_all, "flag_all": flag_all}
