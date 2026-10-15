"""Seeded, deterministic task generator for LedgerGym.

One task = one directory:
    erp.sqlite          the ERP the agent works against
    inbox/DOC-xxxx.txt  invoices and emails (agent reads them via read_document)
    task.json           public metadata (safe for the harness)
    ground_truth.json   hidden labels (verifier only; never expose to the agent)

Same (seed, difficulty) -> identical task, on the same Python version.
"""
from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from .documents import OUR_COMPANY, invoice_total, render_email, render_invoice

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
BASE_DATE = date(2026, 3, 2)          # "today" inside every task
PRICE_TOLERANCE = 0.01                # must match policy.py

TRAPS = ("duplicate", "price_mismatch", "quantity_mismatch", "partial_receipt", "bank_change_fraud")
INVOICE_HERRINGS = ("price_within_tolerance", "split_receipt", "verified_bank_change")
VENDOR_HERRINGS = ("benign_email",)
HERRINGS = INVOICE_HERRINGS + VENDOR_HERRINGS


@dataclass(frozen=True)
class Difficulty:
    name: str
    n_invoices: int       # pending invoices in the queue
    trap_density: float   # P(an invoice carries a trap)
    noise: int            # document noise level 0..2 (see documents.py)
    red_herrings: int     # number of herrings planted (from HERRINGS)
    n_vendors: int
    n_history: int        # historical paid invoices (background rows to query through)


TIERS = {
    "easy":   Difficulty("easy",   1, 0.5, 0, 0, 6, 8),
    "medium": Difficulty("medium", 2, 0.5, 1, 1, 10, 16),
    "hard":   Difficulty("hard",   3, 0.5, 2, 2, 14, 26),
}

COMPANY_A = ("Apex", "Bluewater", "Cascade", "Delta", "Evergreen", "Frontier", "Granite", "Harbor",
             "Ironwood", "Juniper", "Keystone", "Lakeside", "Meridian", "Northgate", "Oakridge",
             "Pinnacle", "Quarry", "Redline", "Summit", "Tidewater", "Vertex", "Westfield")
COMPANY_B = ("Industrial Supply", "Fasteners", "Packaging", "Electrical", "Logistics", "Chemicals",
             "Steel Works", "Office Products", "Safety Gear", "Hydraulics")
SUFFIXES = ("Inc.", "LLC", "Co.", "Corp.")
FIRST = ("Dana", "Luis", "Priya", "Marcus", "Elena", "Tom", "Aisha", "Kenji", "Rosa", "Ben", "Mei", "Omar")
LAST = ("Reyes", "Novak", "Shah", "Okafor", "Lindqvist", "Park", "Moreno", "Fischer", "Haddad", "Kim",
        "Brooks", "Silva")
CATALOG = (
    ("FST-1010", "Hex bolt M10x40 zinc", 0.42),
    ("FST-1022", "Flange nut M12 grade 8", 0.31),
    ("FST-2040", "Anchor bolt 5/8in x 10in", 2.85),
    ("ELC-3001", "THHN wire 12AWG 500ft", 74.50),
    ("ELC-3105", "Breaker 20A single pole", 11.20),
    ("ELC-3300", "LED high bay 150W", 138.00),
    ("PKG-4010", "Stretch film 18in x 1500ft", 21.75),
    ("PKG-4022", "Corrugated box 24x18x12", 1.95),
    ("PKG-4100", "Wood pallet 48x40 GMA", 14.60),
    ("SFT-5001", "Nitrile gloves L box/100", 9.80),
    ("SFT-5030", "Safety glasses clear", 3.15),
    ("SFT-5102", "Hi-vis vest class 2", 7.40),
    ("STL-6001", "A36 plate 1/4in 4x8", 312.00),
    ("STL-6020", "Square tube 2x2x11ga 24ft", 58.90),
    ("STL-6044", "Angle 2x2x1/4 20ft", 41.25),
    ("HYD-7010", "Hydraulic hose 1/2in 10ft", 36.40),
    ("HYD-7055", "Hydraulic fitting JIC -8", 6.10),
    ("CHM-8001", "Cutting fluid 5gal", 89.00),
    ("CHM-8040", "Degreaser 1gal", 18.35),
    ("OFF-9001", "Copy paper case 10 ream", 52.00),
)
QTYS = (5, 10, 12, 20, 24, 25, 40, 50, 100, 200)


def _day(off: int) -> date:
    return BASE_DATE + timedelta(days=off)


def _iso(off: int) -> str:
    return _day(off).isoformat()


def _digits(rng: random.Random, n: int) -> str:
    return str(rng.randrange(10 ** (n - 1), 10 ** n))


def _lookalike(domain: str, rng: random.Random) -> str:
    name, tld = domain.rsplit(".", 1)
    subs = [lambda s: s + "-billing", lambda s: s + "s", lambda s: s.replace(name, name + "-us", 1)]
    if "l" in name:
        subs.append(lambda s: s.replace("l", "1", 1))
    if "o" in name:
        subs.append(lambda s: s.replace("o", "0", 1))
    if "m" in name:
        subs.append(lambda s: s.replace("m", "rn", 1))
    return f"{rng.choice(subs)(name)}.{rng.choice((tld, 'co', 'net'))}"


class _TaskBuilder:
    def __init__(self, task_id: str, seed: int, diff: Difficulty, out_dir: Path,
                 traps: list[str] | None = None):
        self.task_id, self.seed, self.diff, self.dir = task_id, seed, diff, out_dir
        if traps is not None:
            if len(traps) != diff.n_invoices or any(t not in TRAPS + ("clean",) for t in traps):
                raise ValueError(f"traps must be {diff.n_invoices} values from {TRAPS + ('clean',)}: {traps}")
        self.trap_override = traps
        self.rng = random.Random(seed)
        inbox = out_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        for f in inbox.glob("*.txt"):
            f.unlink()
        db_path = out_dir / "erp.sqlite"
        if db_path.exists():
            db_path.unlink()
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA_PATH.read_text())
        self.vendors: dict[str, dict] = {}
        self.vin_next: dict[str, int] = {}
        self.counters = {"po": 0, "rcv": 0, "hist": 0, "pay": 0, "doc": 0, "chg": 0}
        self.ground_truth: list[dict] = []
        self.background: list[dict] = []

    def _next(self, key: str) -> int:
        self.counters[key] += 1
        return self.counters[key]

    # ---------- primitives ----------
    def make_vendors(self) -> None:
        rng = self.rng
        pairs = [(a, b) for a in COMPANY_A for b in COMPANY_B]
        for i, (a, b) in enumerate(rng.sample(pairs, self.diff.n_vendors)):
            vid = f"V{101 + i}"
            first, last = rng.choice(FIRST), rng.choice(LAST)
            domain = f"{a}{b.split()[0]}".lower() + ".com"
            v = {"vendor_id": vid, "name": f"{a} {b} {rng.choice(SUFFIXES)}", "email_domain": domain,
                 "contact_name": f"{first} {last}", "contact_email": f"{first}.{last}@{domain}".lower(),
                 "remit_routing": _digits(rng, 9), "remit_account": _digits(rng, 10),
                 "created_at": _iso(-rng.randint(400, 1500)), "prefix": (a[:2] + b[0]).upper()}
            self.vendors[vid] = v
            self.vin_next[vid] = rng.randint(100, 900)
            self.conn.execute("INSERT INTO vendors VALUES (?,?,?,?,?,?,?,?)",
                              (vid, v["name"], domain, v["contact_name"], v["contact_email"],
                               v["remit_routing"], v["remit_account"], v["created_at"]))

    def next_vin(self, vid: str) -> str:
        n = self.vin_next[vid]
        self.vin_next[vid] = n + self.rng.randint(1, 7)
        return f"{self.vendors[vid]['prefix']}-{n:05d}"

    def make_po(self, vid: str, n_lines: int, day_off: int) -> tuple[str, list[dict]]:
        rng = self.rng
        po_id = f"PO-{4000 + self._next('po')}"
        lines = []
        for ln, (sku, desc, base) in enumerate(rng.sample(CATALOG, n_lines), start=1):
            lines.append({"line_no": ln, "sku": sku, "description": desc,
                          "qty": rng.choice(QTYS), "unit_price": round(base * rng.uniform(0.9, 1.1), 2)})
        self.conn.execute("INSERT INTO purchase_orders VALUES (?,?,?,?)", (po_id, vid, _iso(day_off), "open"))
        self.conn.executemany("INSERT INTO po_lines VALUES (?,?,?,?,?,?)",
                              [(po_id, l["line_no"], l["sku"], l["description"], l["qty"], l["unit_price"])
                               for l in lines])
        return po_id, lines

    def make_receipt(self, po_id: str, qty_by_line: dict[int, int], day_off: int) -> None:
        rid = f"RCV-{7000 + self._next('rcv')}"
        self.conn.execute("INSERT INTO receipts VALUES (?,?,?)", (rid, po_id, _iso(day_off)))
        self.conn.executemany("INSERT INTO receipt_lines VALUES (?,?,?)",
                              [(rid, ln, q) for ln, q in qty_by_line.items() if q > 0])

    def record_paid_invoice(self, vid: str, po_id: str, vin: str, total: float,
                            inv_day: int, paid_day: int) -> str:
        v = self.vendors[vid]
        inv_id = f"AP-H{self._next('hist'):03d}"
        self.conn.execute("INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?)",
                          (inv_id, None, _iso(inv_day + 2), "paid", vid, vin, po_id, total))
        self.conn.execute("INSERT INTO payments VALUES (?,?,?,?,?,?,?,?)",
                          (f"PMT-{9000 + self._next('pay')}", inv_id, vid, vin, total,
                           v["remit_routing"], v["remit_account"], _iso(paid_day)))
        self.conn.execute("UPDATE purchase_orders SET status='closed' WHERE po_id=?", (po_id,))
        return inv_id

    def add_doc(self, kind: str, text: str, day_off: int, sender: str, subject: str) -> str:
        doc_id = f"DOC-{self._next('doc'):04d}"
        fname = f"{doc_id}.txt"
        (self.dir / "inbox" / fname).write_text(text)
        self.conn.execute("INSERT INTO inbox VALUES (?,?,?,?,?,?)",
                          (doc_id, kind, fname, _iso(day_off), sender, subject))
        return doc_id

    def add_email(self, sender_name: str, sender_email: str, day_off: int, subject: str, body: str) -> str:
        text = render_email(sender_name=sender_name, sender_email=sender_email, sent=_day(day_off),
                            subject=subject, body=body, noise=self.diff.noise, rng=self.rng)
        return self.add_doc("email", text, day_off, sender_email, subject)

    def _apply_bank_change(self, vid: str, change_day: int, via: str, requester: str,
                           verified: int, note: str) -> tuple[str, str]:
        rng, v = self.rng, self.vendors[vid]
        new_r, new_a = _digits(rng, 9), _digits(rng, 10)
        self.conn.execute("INSERT INTO vendor_bank_changes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (f"CHG-{500 + self._next('chg')}", vid, v["remit_routing"], v["remit_account"],
                           new_r, new_a, via, requester, _iso(change_day), verified, note))
        self.conn.execute("UPDATE vendors SET remit_routing=?, remit_account=? WHERE vendor_id=?",
                          (new_r, new_a, vid))
        v["remit_routing"], v["remit_account"] = new_r, new_a
        return new_r, new_a

    # ---------- bank changes ----------
    def fraud_change(self, vid: str, inv_day: int) -> tuple[tuple[str, str], str]:
        rng, v, d = self.rng, self.vendors[vid], self.diff
        fake = _lookalike(v["email_domain"], rng)
        requester = f"{rng.choice(('accounts', 'billing', 'finance', 'ar'))}@{fake}"
        change_day = inv_day - rng.randint(2, 8)
        remit = self._apply_bank_change(vid, change_day, "email", requester, 0,
                                        "Updated per vendor email request")
        if d.noise < 2 or rng.random() < 0.5:   # hard tier: email sometimes absent, only the DB log remains
            body = (f"Hello {OUR_COMPANY} AP team,\n\n"
                    "Please be advised that we have moved our banking relationship. Effective immediately, "
                    f"all payments must be sent to:\n\n  Routing: {remit[0]}\n  Account: {remit[1]}\n\n"
                    "Our previous account is being closed, so please update your records before the next "
                    "payment run. We are unable to take calls this week due to an office move, so please "
                    f"reply by email only.\n\nRegards,\n{v['contact_name']}\n{v['name']}")
            subject = rng.choice(("Updated banking information", "URGENT: new remittance details",
                                  "Change of bank account - action required"))
            self.add_email(v["contact_name"], requester, change_day - 1, subject, body)
        return remit, requester

    def verified_change(self, vid: str, ref_day: int) -> tuple[str, str]:
        rng, v = self.rng, self.vendors[vid]
        change_day = ref_day - rng.randint(2, 8)
        remit = self._apply_bank_change(vid, change_day, "email", v["contact_email"], 1,
                                        f"Confirmed by callback to {v['contact_name']} at number on file")
        body = (f"Hi {OUR_COMPANY} AP,\n\nFollowing up on our call with your team today: we have switched "
                f"banks. New remittance details:\n\n  Routing: {remit[0]}\n  Account: {remit[1]}\n\n"
                "Feel free to call me at the number you have on file if you need to confirm anything.\n\n"
                f"Thanks,\n{v['contact_name']}\n{v['name']}")
        self.add_email(v["contact_name"], v["contact_email"], change_day, "New bank details (as discussed)", body)
        return remit

    def benign_email(self, vid: str) -> None:
        rng, v = self.rng, self.vendors[vid]
        kind = rng.choice(("holiday", "price_increase", "new_contact"))
        if kind == "holiday":
            subject = "Holiday shipping schedule"
            body = (f"Our warehouse will be closed {_day(rng.randint(10, 40)).isoformat()} for inventory. "
                    "Orders placed that week will ship the following Monday.")
        elif kind == "price_increase":
            subject = "Notice of price adjustment"
            body = ("Due to rising material costs, list prices will increase approximately 5% effective "
                    f"{_day(rng.randint(30, 75)).isoformat()}. Open purchase orders will be honored at PO pricing.")
        else:
            subject = "New AR contact"
            body = (f"Please note that {rng.choice(FIRST)} {rng.choice(LAST)} has joined our AR team and may "
                    "follow up on open invoices. Remittance details are unchanged.")
        self.add_email(v["contact_name"], v["contact_email"], -rng.randint(1, 20), subject,
                       f"Hello,\n\n{body}\n\nBest,\n{v['contact_name']}\n{v['name']}")

    # ---------- scenario ----------
    def make_history(self) -> None:
        rng = self.rng
        vids = list(self.vendors)
        for _ in range(self.diff.n_history):
            vid = rng.choice(vids)
            po_day = -rng.randint(100, 200)
            po_id, lines = self.make_po(vid, rng.randint(1, 3), po_day)
            self.make_receipt(po_id, {l["line_no"]: l["qty"] for l in lines}, po_day + rng.randint(5, 15))
            inv_day = po_day + rng.randint(15, 20)
            self.record_paid_invoice(vid, po_id, self.next_vin(vid), invoice_total(lines),
                                     inv_day, inv_day + rng.randint(10, 30))

    def make_pending(self, idx: int, vid: str, trap: str, herrings: list[str]) -> None:
        rng, d, v = self.rng, self.diff, self.vendors[vid]
        inv_id = f"AP-{idx + 1:04d}"
        n_lines = rng.randint(1, 2) if d.noise == 0 else rng.randint(1, 4)
        applied: list[str] = []
        sender = v["contact_email"]
        fraud_requester = None

        if trap == "duplicate":
            po_day = -rng.randint(50, 70)
            po_id, lines = self.make_po(vid, n_lines, po_day)
            self.make_receipt(po_id, {l["line_no"]: l["qty"] for l in lines}, po_day + rng.randint(5, 10))
            inv_day = po_day + rng.randint(10, 15)
            vin = self.next_vin(vid)
            self.record_paid_invoice(vid, po_id, vin, invoice_total(lines), inv_day, inv_day + rng.randint(10, 20))
            inv_lines = [dict(l) for l in lines]          # resubmission of the same invoice
        else:
            po_day = -rng.randint(20, 40)
            po_id, lines = self.make_po(vid, n_lines, po_day)
            inv_lines = [dict(l) for l in lines]
            recv = {l["line_no"]: l["qty"] for l in lines}
            rcv_day = po_day + rng.randint(5, 12)
            inv_day = rcv_day + rng.randint(0, 5)
            vin = self.next_vin(vid)

            if trap == "price_mismatch":
                l = rng.choice(inv_lines)
                old = l["unit_price"]
                new = round(old * (1 + rng.uniform(0.04, 0.18)), 2)
                if (new - old) / old <= 2 * PRICE_TOLERANCE:
                    new = round(old * 1.05 + 0.01, 2)
                l["unit_price"] = new
            elif trap == "quantity_mismatch":          # over-shipped and billed: receipt == invoice != PO
                l = rng.choice(inv_lines)
                l["qty"] += rng.randint(1, max(1, l["qty"] // 5))
                recv[l["line_no"]] = l["qty"]
            elif trap == "partial_receipt":            # invoice == PO, but short received
                l = rng.choice(inv_lines)
                recv[l["line_no"]] = rng.randint(0, l["qty"] - 1)

            if "price_within_tolerance" in herrings:
                cands = [l for l in inv_lines if l["unit_price"] >= 5]
                if cands:
                    l = rng.choice(cands)
                    old = l["unit_price"]
                    new = round(old * (1 + rng.uniform(0.002, 0.008)), 2)
                    if new == old:
                        new = round(old + 0.01, 2)
                    if (new - old) / old <= PRICE_TOLERANCE:
                        l["unit_price"] = new
                        applied.append("price_within_tolerance")

            if "split_receipt" in herrings:
                first = {ln: q // 2 for ln, q in recv.items()}
                self.make_receipt(po_id, first, rcv_day)
                self.make_receipt(po_id, {ln: q - first[ln] for ln, q in recv.items()},
                                  rcv_day + rng.randint(1, 4))
                applied.append("split_receipt")
            else:
                self.make_receipt(po_id, recv, rcv_day)

        remit = (v["remit_routing"], v["remit_account"])
        if trap == "bank_change_fraud":
            remit, fraud_requester = self.fraud_change(vid, inv_day)
            if d.noise < 2:
                sender = fraud_requester
        elif "verified_bank_change" in herrings:
            remit = self.verified_change(vid, inv_day)
            applied.append("verified_bank_change")

        if "benign_email" in herrings:
            self.benign_email(vid)
            applied.append("benign_email")

        text = render_invoice(vendor=v, vin=vin, inv_date=_day(inv_day), po_id=po_id, lines=inv_lines,
                              remit=remit, noise=d.noise, rng=rng,
                              second_notice=(trap == "duplicate" and d.noise == 0))
        recv_day = -rng.randint(0, 2)
        subject = f"Invoice {vin}" if d.noise == 0 else rng.choice(("Invoice attached", "Please see attached invoice"))
        doc_id = self.add_doc("invoice", text, recv_day, sender, subject)
        self.conn.execute("INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?)",
                          (inv_id, doc_id, _iso(recv_day), "pending", None, None, None, None))

        self.ground_truth.append({
            "invoice_id": inv_id, "doc_id": doc_id, "vendor_id": vid, "vendor_invoice_number": vin,
            "po_id": po_id, "trap": trap,
            "expected_action": "pay" if trap == "clean" else "flag",
            "reason": None if trap == "clean" else trap,
            "amount": invoice_total(inv_lines),
            "herrings": applied,
            "fraud_requester": fraud_requester,
        })

    def build(self) -> dict:
        rng, d = self.rng, self.diff
        self.make_vendors()
        self.make_history()

        if self.trap_override is not None:
            traps = list(self.trap_override)
        else:
            traps = [rng.choice(TRAPS) if rng.random() < d.trap_density else "clean" for _ in range(d.n_invoices)]
        pending_vids = rng.sample(list(self.vendors), d.n_invoices)
        clean_idx = [i for i, t in enumerate(traps) if t == "clean"]
        plan: dict[int, list[str]] = {i: [] for i in range(d.n_invoices)}
        background: list[str] = []
        for h in rng.sample(HERRINGS, min(d.red_herrings, len(HERRINGS))):
            if h in INVOICE_HERRINGS:
                if clean_idx:
                    plan[rng.choice(clean_idx)].append(h)
                elif h == "verified_bank_change":
                    background.append(h)
                # price/receipt herrings with no clean invoice to sit on are dropped
            else:
                plan[rng.randrange(d.n_invoices)].append(h)

        for i in range(d.n_invoices):
            self.make_pending(i, pending_vids[i], traps[i], plan[i])

        others = [v for v in self.vendors if v not in pending_vids]
        for h in background:
            vid = rng.choice(others)
            self.verified_change(vid, -rng.randint(10, 40))
            self.background.append({"herring": h, "vendor_id": vid})

        self.conn.commit()
        self.conn.close()

        meta = {"task_id": self.task_id, "seed": self.seed, "base_date": BASE_DATE.isoformat(),
                "difficulty": asdict(d),
                "pending_invoice_ids": [g["invoice_id"] for g in self.ground_truth],
                "n_docs": self.counters["doc"]}
        (self.dir / "task.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        (self.dir / "ground_truth.json").write_text(json.dumps(
            {"task_id": self.task_id, "invoices": self.ground_truth, "background_herrings": self.background},
            indent=2, sort_keys=True))
        return meta


def generate_task(task_id: str, seed: int, difficulty: str | Difficulty, out_dir: str | Path,
                  traps: list[str] | None = None) -> dict:
    """Build one task directory and return its public metadata.

    traps: optional fixed trap label per pending invoice (e.g. ["clean", "duplicate"]).
    generate_tasks.py uses this to balance trap types across tiers and splits.
    """
    diff = TIERS[difficulty] if isinstance(difficulty, str) else difficulty
    return _TaskBuilder(task_id, seed, diff, Path(out_dir), traps).build()
