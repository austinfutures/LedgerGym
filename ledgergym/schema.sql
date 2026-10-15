-- LedgerGym ERP schema. One SQLite file per task.

CREATE TABLE vendors (
    vendor_id      TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    email_domain   TEXT NOT NULL,
    contact_name   TEXT NOT NULL,
    contact_email  TEXT NOT NULL,
    remit_routing  TEXT NOT NULL,   -- current bank details on file
    remit_account  TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

-- Audit log of bank-detail changes applied to the vendor master.
CREATE TABLE vendor_bank_changes (
    change_id             TEXT PRIMARY KEY,
    vendor_id             TEXT NOT NULL REFERENCES vendors(vendor_id),
    old_routing           TEXT,
    old_account           TEXT,
    new_routing           TEXT,
    new_account           TEXT,
    requested_via         TEXT NOT NULL,     -- 'email' | 'portal' | 'phone'
    requester_email       TEXT NOT NULL,
    changed_at            TEXT NOT NULL,
    verified_by_callback  INTEGER NOT NULL,  -- 1 = confirmed via phone number on file
    note                  TEXT
);

CREATE TABLE purchase_orders (
    po_id       TEXT PRIMARY KEY,
    vendor_id   TEXT NOT NULL REFERENCES vendors(vendor_id),
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL            -- 'open' | 'closed'
);

CREATE TABLE po_lines (
    po_id        TEXT NOT NULL REFERENCES purchase_orders(po_id),
    line_no      INTEGER NOT NULL,
    sku          TEXT NOT NULL,
    description  TEXT NOT NULL,
    qty_ordered  INTEGER NOT NULL,
    unit_price   REAL NOT NULL,
    PRIMARY KEY (po_id, line_no)
);

CREATE TABLE receipts (
    receipt_id   TEXT PRIMARY KEY,
    po_id        TEXT NOT NULL REFERENCES purchase_orders(po_id),
    received_at  TEXT NOT NULL
);

CREATE TABLE receipt_lines (
    receipt_id    TEXT NOT NULL REFERENCES receipts(receipt_id),
    line_no       INTEGER NOT NULL,      -- matches po_lines.line_no
    qty_received  INTEGER NOT NULL,
    PRIMARY KEY (receipt_id, line_no)
);

-- Documents waiting in the AP inbox (invoices and emails).
CREATE TABLE inbox (
    doc_id       TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('invoice', 'email')),
    filename     TEXT NOT NULL,
    received_at  TEXT NOT NULL,
    sender       TEXT NOT NULL,
    subject      TEXT NOT NULL
);

-- AP queue. Pending rows only know their document; the clerk extracts the rest.
CREATE TABLE invoices (
    invoice_id             TEXT PRIMARY KEY,
    doc_id                 TEXT REFERENCES inbox(doc_id),
    received_at            TEXT NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ('pending', 'paid', 'flagged')),
    vendor_id              TEXT,
    vendor_invoice_number  TEXT,
    po_id                  TEXT,
    total                  REAL
);

CREATE TABLE payments (
    payment_id             TEXT PRIMARY KEY,
    invoice_id             TEXT NOT NULL,
    vendor_id              TEXT NOT NULL,
    vendor_invoice_number  TEXT,
    amount                 REAL NOT NULL,
    remit_routing          TEXT NOT NULL,
    remit_account          TEXT NOT NULL,
    paid_at                TEXT NOT NULL
);

CREATE TABLE exceptions (
    exception_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id    TEXT NOT NULL,
    reason_code   TEXT NOT NULL,
    note          TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE outbox (
    email_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    to_address  TEXT NOT NULL,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    sent_at     TEXT NOT NULL
);
