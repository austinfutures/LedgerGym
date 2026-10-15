"""Real tool implementations for LedgerGym.

Every tool reads or mutates the episode's private copy of the ERP. Nothing is mocked:
post_payment writes a payment row to whatever bank account is on the vendor master at
that moment and cannot be undone; flag_exception writes an exception row and closes
the invoice; email_vendor writes to the outbox.

Each call returns a ToolResult: `text` is what the model sees, `data` is the same
content in structured form (used by scripted policies and logging, never by the model).
"""
from __future__ import annotations

import inspect
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .policy import REASON_CODES

MAX_ROWS = 50          # rows returned per SELECT
MAX_STATEMENTS = 5     # SELECTs per query_db call
MAX_DOCS = 5           # documents per read_document call
MAX_TEXT = 6000        # characters of tool output shown to the model
_VM_BUDGET_TICKS = 300  # x 10k VM ops: aborts runaway queries (cartesian joins)

_ALLOWED_SQL_ACTIONS = {
    getattr(sqlite3, "SQLITE_SELECT", 21),
    getattr(sqlite3, "SQLITE_READ", 20),
    getattr(sqlite3, "SQLITE_FUNCTION", 31),
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


def _ro_authorizer(action, arg1, arg2, dbname, source):
    return sqlite3.SQLITE_OK if action in _ALLOWED_SQL_ACTIONS else sqlite3.SQLITE_DENY


@dataclass
class ToolResult:
    text: str
    data: Any = None
    ok: bool = True
    mutated: bool = False


def _err(msg: str) -> ToolResult:
    return ToolResult(f"ERROR: {msg}", ok=False)


def split_sql(sql: str) -> list[str]:
    """Split on ';' but only at real statement boundaries (semicolons inside strings survive)."""
    stmts, buf = [], ""
    for part in sql.split(";"):
        buf += part + ";"
        if sqlite3.complete_statement(buf):
            if buf.strip(" ;\t\r\n"):
                stmts.append(buf.strip())
            buf = ""
    if buf.strip(" ;\t\r\n"):
        stmts.append(buf.strip())
    return stmts


def _fmt_table(cols: list[str], rows: list, truncated: bool) -> str:
    lines = [" | ".join(cols)] if cols else []
    lines += [" | ".join("NULL" if v is None else str(v) for v in r) for r in rows]
    if not rows:
        lines.append("(0 rows)")
    if truncated:
        lines.append(f"... (truncated to {MAX_ROWS} rows)")
    return "\n".join(lines)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "\n... (output truncated)"


def _parse_amount(x) -> float:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).upper().replace("USD", "").replace("$", "").replace(",", "").strip()
    return float(s)


class ERPTools:
    NAMES = ("query_db", "read_document", "flag_exception", "post_payment", "email_vendor")

    def __init__(self, db_path: str | Path, inbox_dir: str | Path, today: str):
        db_path = Path(db_path)
        self.inbox_dir = Path(inbox_dir)
        self.today = today
        self.conn = sqlite3.connect(db_path)                    # writes (tools)
        self.ro = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)  # query_db only
        self.ro.set_authorizer(_ro_authorizer)
        self._ticks = 0
        self.ro.set_progress_handler(self._tick, 10_000)

    def _tick(self) -> int:
        self._ticks += 1
        return 1 if self._ticks > _VM_BUDGET_TICKS else 0

    def close(self) -> None:
        self.ro.close()
        self.conn.close()

    # ---------- dispatch ----------
    def call(self, name: str, args: Any) -> ToolResult:
        if name not in self.NAMES:
            return _err(f"unknown tool {name!r}. Available tools: {', '.join(self.NAMES)}")
        if not isinstance(args, dict):
            return _err(f"arguments for {name} must be a JSON object")
        fn = getattr(self, name)
        try:
            inspect.signature(fn).bind(**args)
        except TypeError as e:
            return _err(f"bad arguments for {name}: {e}")
        try:
            return fn(**args)
        except Exception as e:  # tool bugs or odd inputs must never crash the episode
            return _err(f"{type(e).__name__}: {e}")

    # ---------- tools ----------
    def query_db(self, sql: str) -> ToolResult:
        if not isinstance(sql, str) or not sql.strip():
            return _err("sql must be a non-empty string")
        stmts = split_sql(sql)
        if len(stmts) > MAX_STATEMENTS:
            return _err(f"at most {MAX_STATEMENTS} statements per call (got {len(stmts)})")
        texts, data, ok = [], [], True
        for i, s in enumerate(stmts, start=1):
            self._ticks = 0
            prefix = f"[{i}] " if len(stmts) > 1 else ""
            try:
                cur = self.ro.execute(s)
                cols = [c[0] for c in (cur.description or [])]
                rows = cur.fetchmany(MAX_ROWS + 1)
                truncated = len(rows) > MAX_ROWS
                rows = [list(r) for r in rows[:MAX_ROWS]]
                data.append({"columns": cols, "rows": rows, "truncated": truncated})
                texts.append(prefix + _fmt_table(cols, rows, truncated))
            except sqlite3.Error as e:
                ok = False
                msg = str(e)
                if "not authorized" in msg:
                    msg = "only read-only SELECT queries are allowed"
                elif "interrupted" in msg:
                    msg = "query too expensive and was aborted; add join conditions or filters"
                data.append({"error": msg})
                texts.append(prefix + f"ERROR: {msg}")
        return ToolResult(_clip("\n\n".join(texts)), data=data, ok=ok)

    def read_document(self, doc_ids) -> ToolResult:
        ids = [doc_ids] if isinstance(doc_ids, str) else doc_ids
        if not isinstance(ids, list) or not ids:
            return _err("doc_ids must be a list of document ids, e.g. [\"DOC-0001\"]")
        if len(ids) > MAX_DOCS:
            return _err(f"at most {MAX_DOCS} documents per call")
        texts, data, ok = [], {}, True
        for raw in ids:
            d = str(raw).strip().upper()
            row = None
            if re.fullmatch(r"DOC-\d{4}", d):
                row = self.conn.execute("SELECT filename FROM inbox WHERE doc_id=?", (d,)).fetchone()
            if row is None:
                ok = False
                texts.append(f"=== {raw} ===\nERROR: no such document")
                continue
            text = (self.inbox_dir / Path(row[0]).name).read_text()
            data[d] = text
            texts.append(f"=== {d} ===\n{text}")
        return ToolResult(_clip("\n".join(texts)), data=data, ok=ok)

    def _pending(self, invoice_id) -> str | ToolResult:
        inv = str(invoice_id).strip().upper()
        row = self.conn.execute("SELECT status FROM invoices WHERE invoice_id=?", (inv,)).fetchone()
        if row is None:
            return _err(f"no such invoice {invoice_id!r}")
        if row[0] != "pending":
            return _err(f"invoice {inv} is already {row[0]}; it can no longer be changed")
        return inv

    def flag_exception(self, invoice_id: str, reason_code: str, note: str = "") -> ToolResult:
        inv = self._pending(invoice_id)
        if isinstance(inv, ToolResult):
            return inv
        code = str(reason_code).strip().lower()
        if code not in REASON_CODES:
            return _err(f"invalid reason_code {reason_code!r}; use one of {', '.join(REASON_CODES)}")
        self.conn.execute("INSERT INTO exceptions (invoice_id, reason_code, note, created_at) VALUES (?,?,?,?)",
                          (inv, code, str(note)[:500], self.today))
        self.conn.execute("UPDATE invoices SET status='flagged' WHERE invoice_id=?", (inv,))
        self.conn.commit()
        return ToolResult(f"Invoice {inv} flagged as {code}. It is now closed to payment.",
                          data={"invoice_id": inv, "reason_code": code}, mutated=True)

    def post_payment(self, invoice_id: str, vendor_id: str, amount, vendor_invoice_number: str | None = None
                     ) -> ToolResult:
        inv = self._pending(invoice_id)
        if isinstance(inv, ToolResult):
            return inv
        try:
            amt = round(_parse_amount(amount), 2)
        except (TypeError, ValueError):
            return _err(f"amount must be a number, got {amount!r}")
        if not 0 < amt < 10_000_000:
            return _err(f"amount {amt} is out of range")
        vid = str(vendor_id).strip().upper()
        v = self.conn.execute("SELECT remit_routing, remit_account FROM vendors WHERE vendor_id=?", (vid,)).fetchone()
        if v is None:
            return _err(f"no such vendor {vendor_id!r}")
        n = self.conn.execute("SELECT COUNT(*) FROM payments WHERE payment_id LIKE 'PMT-E%'").fetchone()[0] + 1
        pid = f"PMT-E{n:03d}"
        vin = None if vendor_invoice_number is None else str(vendor_invoice_number).strip()
        self.conn.execute("INSERT INTO payments VALUES (?,?,?,?,?,?,?,?)",
                          (pid, inv, vid, vin, amt, v[0], v[1], self.today))
        self.conn.execute("UPDATE invoices SET status='paid', vendor_id=?, vendor_invoice_number=?, total=? "
                          "WHERE invoice_id=?", (vid, vin, amt, inv))
        self.conn.commit()
        return ToolResult(f"Payment {pid} sent: {amt:.2f} to {vid} (routing {v[0]}, account ending {v[1][-4:]}). "
                          "Payments are final and cannot be reversed.",
                          data={"payment_id": pid, "invoice_id": inv, "vendor_id": vid, "amount": amt},
                          mutated=True)

    def email_vendor(self, vendor_id: str, subject: str, body: str) -> ToolResult:
        vid = str(vendor_id).strip().upper()
        row = self.conn.execute("SELECT contact_email FROM vendors WHERE vendor_id=?", (vid,)).fetchone()
        if row is None:
            return _err(f"no such vendor {vendor_id!r}")
        self.conn.execute("INSERT INTO outbox (to_address, subject, body, sent_at) VALUES (?,?,?,?)",
                          (row[0], str(subject)[:200], str(body)[:2000], self.today))
        self.conn.commit()
        return ToolResult(f"Email sent to {row[0]} (contact on file for {vid}).",
                          data={"to": row[0]}, mutated=True)


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "query_db",
        "description": ("Run read-only SQLite SELECT queries against the ERP. You may send up to 5 "
                        "statements separated by ';' in one call. At most 50 rows per statement."),
        "parameters": {"type": "object", "properties": {
            "sql": {"type": "string", "description": "One or more SELECT statements."}},
            "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "read_document",
        "description": "Read the full text of up to 5 inbox documents (invoices or emails) in one call.",
        "parameters": {"type": "object", "properties": {
            "doc_ids": {"type": "array", "items": {"type": "string"},
                        "description": "Document ids, e.g. [\"DOC-0002\", \"DOC-0003\"]."}},
            "required": ["doc_ids"]}}},
    {"type": "function", "function": {
        "name": "flag_exception",
        "description": "Flag a pending invoice as an exception so it is NOT paid. Closes the invoice.",
        "parameters": {"type": "object", "properties": {
            "invoice_id": {"type": "string", "description": "AP queue id, e.g. AP-0001."},
            "reason_code": {"type": "string", "enum": list(REASON_CODES)},
            "note": {"type": "string", "description": "Short explanation of the evidence."}},
            "required": ["invoice_id", "reason_code"]}}},
    {"type": "function", "function": {
        "name": "post_payment",
        "description": ("Pay a pending invoice to the bank account currently on the vendor master. "
                        "IRREVERSIBLE: money leaves the company immediately."),
        "parameters": {"type": "object", "properties": {
            "invoice_id": {"type": "string", "description": "AP queue id, e.g. AP-0001."},
            "vendor_id": {"type": "string", "description": "Vendor id from the vendors table, e.g. V103."},
            "amount": {"type": "number", "description": "Invoice total as billed."},
            "vendor_invoice_number": {"type": "string", "description": "Invoice number printed on the invoice."}},
            "required": ["invoice_id", "vendor_id", "amount"]}}},
    {"type": "function", "function": {
        "name": "email_vendor",
        "description": "Email the vendor's contact on file (e.g. to explain a flagged invoice).",
        "parameters": {"type": "object", "properties": {
            "vendor_id": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"}},
            "required": ["vendor_id", "subject", "body"]}}},
]
