"""Rendering of invoices and emails as text documents, with controllable noise.

noise 0: canonical formatting (ISO dates, plain numbers, exact invoice numbers)
noise 1: varied date / money formats, vendor-name casing, PO label variants
noise 2: + reordered line items, typos in descriptions, invoice-number formatting
         variants, hyphenated account numbers, boilerplate
Numbers that matter for matching (qty, prices, totals, IDs' digits) are never corrupted.
"""
from __future__ import annotations

import random
from datetime import date

OUR_COMPANY = "Northwind Fabrication"
AP_EMAIL = "payables@northwindfab.com"

DATE_FMTS = ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y", "%B %d, %Y")
BOILERPLATE = (
    "Thank you for your business!",
    "Please reference the invoice number on your remittance.",
    "Late payments are subject to a 1.5% monthly service charge.",
    "Questions about this invoice? Contact our AR department.",
    "All goods remain property of seller until paid in full.",
)


def fmt_date(d: date, noise: int, rng: random.Random) -> str:
    return d.isoformat() if noise == 0 else d.strftime(rng.choice(DATE_FMTS))


def fmt_money(x: float, noise: int, rng: random.Random) -> str:
    if noise == 0:
        return f"{x:.2f}"
    return rng.choice((f"${x:,.2f}", f"{x:,.2f} USD", f"USD {x:,.2f}", f"${x:,.2f}"))


def fmt_account(acct: str, noise: int, rng: random.Random) -> str:
    if noise >= 2 and rng.random() < 0.5:
        return f"{acct[:3]}-{acct[3:6]}-{acct[6:]}"
    return acct


def vary_vin(vin: str, noise: int, rng: random.Random) -> str:
    if noise < 2:
        return vin
    return rng.choice((vin, vin.replace("-", " "), vin.replace("-", ""), "#" + vin, vin.lower()))


def vary_name(name: str, noise: int, rng: random.Random) -> str:
    if noise == 0:
        return name
    return rng.choice((name, name.upper(), name.replace(".", ""), name.rsplit(" ", 1)[0]))


def typo(text: str, rng: random.Random) -> str:
    """Swap two adjacent interior letters in one alphabetic word (never touches digits)."""
    words = text.split(" ")
    i = rng.randrange(len(words))
    w = words[i]
    if len(w) >= 4 and w.isalpha():
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def invoice_total(lines: list[dict]) -> float:
    return round(sum(round(l["qty"] * l["unit_price"], 2) for l in lines), 2)


def render_invoice(*, vendor: dict, vin: str, inv_date: date, po_id: str, lines: list[dict],
                   remit: tuple[str, str], noise: int, rng: random.Random,
                   second_notice: bool = False) -> str:
    out: list[str] = []
    if second_notice:
        out.append("*** SECOND NOTICE - PAYMENT NOT YET RECEIVED ***")
    out.append("INVOICE")
    out.append(vary_name(vendor["name"], noise, rng))
    out.append(f"{vendor['contact_name']} | {vendor['contact_email']}")
    out.append("")
    out.append(f"Bill to: {OUR_COMPANY}, Accounts Payable")
    out.append(f"Invoice No: {vary_vin(vin, noise, rng)}")
    out.append(f"Invoice Date: {fmt_date(inv_date, noise, rng)}")
    po_label = "PO" if noise == 0 else rng.choice(("PO", "P.O. #", "Purchase Order", "Your PO Ref"))
    out.append(f"{po_label}: {po_id}")
    out.append("")

    rows = list(lines)
    if noise >= 2:
        rng.shuffle(rows)
    out.append(f"{'SKU':<10} {'Description':<34} {'Qty':>6} {'Unit Price':>14} {'Amount':>14}")
    for l in rows:
        desc = typo(l["description"], rng) if noise >= 2 and rng.random() < 0.3 else l["description"]
        amt = round(l["qty"] * l["unit_price"], 2)
        out.append(f"{l['sku']:<10} {desc:<34} {l['qty']:>6} "
                   f"{fmt_money(l['unit_price'], noise, rng):>14} {fmt_money(amt, noise, rng):>14}")
    out.append("")
    out.append(f"Total Due: {fmt_money(invoice_total(lines), noise, rng)}")
    routing, acct = remit
    out.append(f"Remit via ACH: routing {routing}, account {fmt_account(acct, noise, rng)}")
    out.append("Terms: Net 30")
    if noise >= 2:
        out.append("")
        out.extend(rng.sample(BOILERPLATE, 2))
    return "\n".join(out) + "\n"


def render_email(*, sender_name: str, sender_email: str, sent: date, subject: str, body: str,
                 noise: int, rng: random.Random) -> str:
    return (f"From: {sender_name} <{sender_email}>\n"
            f"To: {AP_EMAIL}\n"
            f"Date: {fmt_date(sent, noise, rng)}\n"
            f"Subject: {subject}\n\n"
            f"{body}\n")
