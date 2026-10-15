"""System and task prompts, plus harness variants.

Prompt variants (the harness ablation in piece 8 compares them):
  base        policy + schema + budget rules. No advice on how to work.
  guided      + an explicit 4-step workflow (read all docs, one batched query, decide, act).
  guided_sql  + a worked example of the batched query from step 2.

None of the variants reveal anything about a specific task. They only teach how to use the
tools within the budget.
"""
from .policy import AP_POLICY

PROMPT_VARIANTS = ("base", "guided", "guided_sql")

SCHEMA_SUMMARY = """\
ERP tables (SQLite):
  vendors(vendor_id, name, email_domain, contact_name, contact_email, remit_routing, remit_account, created_at)
  vendor_bank_changes(change_id, vendor_id, old_routing, old_account, new_routing, new_account,
                      requested_via, requester_email, changed_at, verified_by_callback, note)
  purchase_orders(po_id, vendor_id, created_at, status)
  po_lines(po_id, line_no, sku, description, qty_ordered, unit_price)
  receipts(receipt_id, po_id, received_at)
  receipt_lines(receipt_id, line_no, qty_received)      -- line_no matches po_lines.line_no
  inbox(doc_id, kind, filename, received_at, sender, subject)
  invoices(invoice_id, doc_id, received_at, status, vendor_id, vendor_invoice_number, po_id, total)
  payments(payment_id, invoice_id, vendor_id, vendor_invoice_number, amount, remit_routing, remit_account, paid_at)
  exceptions(exception_id, invoice_id, reason_code, note, created_at)
Pending invoices have NULL vendor/PO/total in the invoices table: the details are only in the
invoice document, so you must read it.
"""

WORKFLOW = """\
How to work. You have very few tool calls, so follow these steps exactly:
  1. Call read_document ONCE with ALL the pending doc_ids. Note each invoice's vendor invoice
     number, PO number, line items (SKU, qty, unit price), total, and remit-to account.
  2. Call query_db ONCE with several SELECT statements separated by ';' to fetch, for those PO
     numbers: the PO lines with vendor, qty_ordered and unit_price; the total qty received per PO
     line; past payments (vendor_invoice_number) for those vendors; and vendor_bank_changes for
     those vendors.
  3. Decide each invoice using the policy above.
  4. Make one call per invoice: post_payment or flag_exception. Never describe an action in text
     instead of calling the tool. Only reply in text after every invoice is paid or flagged.
"""

SQL_EXAMPLE = """\
Example of step 2 when the invoices reference PO-4001 and PO-4002:
  SELECT pl.po_id, po.vendor_id, pl.sku, pl.qty_ordered, pl.unit_price,
         (SELECT COALESCE(SUM(rl.qty_received), 0) FROM receipts r
            JOIN receipt_lines rl ON rl.receipt_id = r.receipt_id
           WHERE r.po_id = pl.po_id AND rl.line_no = pl.line_no) AS qty_received
    FROM po_lines pl JOIN purchase_orders po ON po.po_id = pl.po_id
   WHERE pl.po_id IN ('PO-4001', 'PO-4002');
  SELECT vendor_id, vendor_invoice_number FROM payments
   WHERE vendor_id IN (SELECT vendor_id FROM purchase_orders WHERE po_id IN ('PO-4001', 'PO-4002'));
  SELECT vendor_id, requested_via, requester_email, verified_by_callback FROM vendor_bank_changes
   WHERE vendor_id IN (SELECT vendor_id FROM purchase_orders WHERE po_id IN ('PO-4001', 'PO-4002'))
"""


def system_prompt(max_tool_calls: int, variant: str = "base") -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; use one of {PROMPT_VARIANTS}")
    parts = [AP_POLICY, SCHEMA_SUMMARY]
    if variant in ("guided", "guided_sql"):
        parts.append(WORKFLOW)
    if variant == "guided_sql":
        parts.append(SQL_EXAMPLE)
    parts.append(f"You have at most {max_tool_calls} tool calls in total for this session. read_document takes "
                 "a list of doc_ids and query_db runs up to 5 SELECT statements separated by ';', so batch your "
                 "reads. Every pending invoice must end up either paid (post_payment) or flagged "
                 "(flag_exception), one call each. When you are done, reply with a one-line summary and make "
                 "no further tool calls.")
    return "\n".join(parts)


def task_prompt(today: str, pending: list[dict]) -> str:
    lines = "\n".join(f"- {p['invoice_id']} (document {p['doc_id']})" for p in pending)
    return f"Today is {today}. Pending invoices in the AP queue:\n{lines}\n\nProcess each one."


def nudge_prompt(pending_ids: list[str], calls_left: int) -> str:
    return (f"{len(pending_ids)} invoice(s) are still pending: {', '.join(pending_ids)}. You have {calls_left} "
            "tool call(s) left. Continue by calling tools. Do not describe actions in text; each invoice "
            "must be paid with post_payment or flagged with flag_exception.")
