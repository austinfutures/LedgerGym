"""System and task prompts. Piece 8 (harness ablation) adds variants alongside these defaults."""
from .policy import AP_POLICY

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
"""


def system_prompt(max_tool_calls: int) -> str:
    return (f"{AP_POLICY}\n{SCHEMA_SUMMARY}\n"
            f"You have at most {max_tool_calls} tool calls in total for this session. read_document takes "
            "a list of doc_ids and query_db runs up to 5 SELECT statements separated by ';', so batch your "
            "reads. Every pending invoice must end up either paid (post_payment) or flagged "
            "(flag_exception), one call each. When you are done, reply with a one-line summary and make "
            "no further tool calls.")


def task_prompt(today: str, pending: list[dict]) -> str:
    lines = "\n".join(f"- {p['invoice_id']} (document {p['doc_id']})" for p in pending)
    return f"Today is {today}. Pending invoices in the AP queue:\n{lines}\n\nProcess each one."
