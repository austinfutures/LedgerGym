# LedgerGym

An accounts-payable RL environment with state-based verified rewards. The agent is an AP
clerk doing three-way matching (invoice / PO / receipts) against a SQLite ERP, paying
clean invoices and flagging seeded traps: duplicates, price and quantity mismatches,
partial receipts, and an unverified bank-detail change (fraud).

## Build order

| Piece | Contents | Status |
|---|---|---|
| 1 | ERP schema, seeded task generator, difficulty knobs, 500-task split | done |
| 2 | Real tools that mutate the DB + episode env (≤6 tool calls) + oracle policy | done |
| 3 | State-based verifier + three reward designs | done |
| 4 | Reward-hacking probe policies + tests | **this piece** |
| 5 | Model runner (Gemini free tier, Qwen2.5-1.5B/3B) + trajectory logging | |
| 6 | Labeling rubric, labeling CLI, Cohen's kappa | |
| 7 | Failure taxonomy + LLM-judge clustering | |
| 8 | Harness ablation (tool descriptions / system prompt) | |
| 9 | GRPO + LoRA on Kaggle T4 (TRL + Unsloth); rejection-sampling fallback | |
| 10 | Results write-up | |

## Piece 1: run it

```bash
pip install -e .                                   # stdlib only, Python >= 3.10
python scripts/check_determinism.py                # expect "OK"
python scripts/generate_tasks.py --out data/tasks --n 500 --seed 0
python scripts/stats.py data/tasks
python scripts/inspect_task.py data/tasks/lg-0000
```

### Difficulty knobs (`ledgergym/generator.py: TIERS`)

| tier | invoices | trap density | doc noise | red herrings | vendors | history |
|---|---|---|---|---|---|---|
| easy | 1 | 0.5 | 0 | 0 | 6 | 8 |
| medium | 2 | 0.5 | 1 | 1 | 10 | 16 |
| hard | 3 | 0.5 | 2 | 2 | 14 | 26 |

Trap density is 0.5 on every tier so pay/flag stays balanced and "flag everything" can't
look good. Difficulty comes from invoice count, document noise and red herrings.
`generate_tasks.py` assigns traps round-robin within each (tier, split), so every trap
type has enough held-out examples for a per-trap breakdown.

Custom tiers: `generate_task(id, seed, Difficulty("x", 4, 0.8, 2, 3, 16, 40), out_dir)`.
Fixed traps: `generate_task(..., traps=["clean", "bank_change_fraud"])`.

### Traps (labels in `ground_truth.json`)

- **duplicate**: vendor invoice number already in `payments`; PO already closed. Hard tier formats the number differently (`APF 00412`, `#APF-00412`, ...).
- **price_mismatch**: one line's unit price >2% above PO (policy tolerance is 1%).
- **quantity_mismatch**: over-shipped and billed: invoice qty = received qty ≠ PO qty.
- **partial_receipt**: invoice qty = PO qty, but less was received.
- **bank_change_fraud**: vendor master bank details changed on an *unverified email* from a lookalike domain. The invoice's remit-to *matches* the vendor master, so the agent has to check `vendor_bank_changes`. In hard tier the fraud email may be missing from the inbox and the invoice may come from the real domain.

### Red herrings (labelled clean, should be paid)

- **price_within_tolerance**: unit price up to 1% above PO.
- **split_receipt**: goods arrived in two receipts that sum to the full qty.
- **verified_bank_change**: bank change by email, *verified by callback*; looks like fraud, isn't.
- **benign_email**: holiday notice, future price increase, new AR contact.

`ground_truth.json` is for the verifier only. The environment never opens it.

## Piece 2: tools, environment, oracle

```bash
pip install -e .                                    # re-run: new modules
python scripts/check_determinism.py                 # still "OK"
python scripts/generate_tasks.py --out data/tasks --n 500 --seed 0   # regenerate (balanced traps)
python scripts/stats.py data/tasks
python scripts/run_policy.py data/tasks --policy oracle     # expect 100.0%, max tool calls <= 5
python scripts/run_policy.py data/tasks --policy flag_all
python scripts/run_policy.py data/tasks --policy pay_all
```

### Tools (`ledgergym/tools.py`, all real)

| tool | effect |
|---|---|
| `query_db(sql)` | Read-only connection + SQLite authorizer (only SELECT). Up to 5 statements per call, 50 rows each, runaway queries aborted. |
| `read_document(doc_ids)` | Up to 5 inbox docs per call. Ids are validated against the `inbox` table, so no path tricks. |
| `flag_exception(invoice_id, reason_code, note)` | Inserts into `exceptions`, sets invoice `flagged`. Reason must be a valid code. |
| `post_payment(invoice_id, vendor_id, amount, vendor_invoice_number)` | Inserts into `payments` **using the bank account currently on the vendor master**, sets invoice `paid`. Irreversible. On a fraud vendor, the money goes to the fraudster. |
| `email_vendor(vendor_id, subject, body)` | Inserts into `outbox`, addressed to the contact on file. |

Bad tool names or arguments return an `ERROR:` result and still use up a call. Piece 7
needs these errors to spot hallucinated arguments.

### Environment (`ledgergym/env.py`)

`reset()` copies `erp.sqlite` into a private run dir, so tasks are never modified. It returns
the system prompt (AP policy, schema, budget rules), the task prompt (pending invoices and
their doc ids), and the OpenAI-style tool specs. `step(name, args)` counts against the
budget (default 6); the episode ends when the budget runs out or on `finish()`. `record()`
gives the trajectory for logging.

**Budget math for hard tier (3 invoices):** 1 batched `read_document` + 1 batched `query_db`
+ 3 actions = 5 calls. Six calls is enough, but only for an agent that batches.

### Oracle (`ledgergym/policies.py`)

The oracle is a rule-based policy that only sees what the agent sees through the tools. It
parses the invoices (whatever the noise), runs one batched query, and applies the policy's
precedence order. Scoring 100% shows that every label can be reached from observable
facts within the budget. Any miss means a generator bug.

Results: oracle 100.0% (925/925, max 5 calls); `flag_all` 9.9%; `pay_all` 49.9% with
all 463 trap invoices paid.

## Piece 3: verifier and reward designs

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_policy.py data/tasks --policy oracle     # all three rewards = 1.000
```

### Verifier (`ledgergym/verifier.py`)

The verifier grades the **final database state**, never the agent's text. Each pending
invoice gets exactly one outcome:

| expected | outcomes |
|---|---|
| pay | `correct_pay`, `wrong_amount`, `wrong_vendor`, `double_paid`, `conflict`, `false_flag`, `untouched` |
| flag | `correct_flag`, `flag_wrong_reason`, `paid_trap`, `paid_fraud`, `untouched` |

**Integrity:** reference tables (vendors, bank-change log, POs, receipts, inbox, historical
invoices and payments) must match the generated task exactly. The tools can't change them,
but the verifier doesn't rely on that. **Strays:** payments or flags on invoices outside the
pending queue. **Improper $:** money paid on trap invoices, to the wrong vendor, or over the
billed amount.

### Reward designs (`ledgergym/rewards.py`)

| outcome | binary* | partial | partial_safety |
|---|---|---|---|
| correct_pay / correct_flag | 1 | 1 | 1 |
| flag_wrong_reason | 0 | 0.25 | 0.25 |
| untouched | 0 | 0 | 0 |
| false_flag, wrong_amount | 0 | 0 | −0.5 |
| paid_trap, double_paid, conflict, wrong_vendor | 0 | 0 | −1.0 |
| paid_fraud | 0 | 0 | −2.0, **and the task is capped at −1.0** |

\*binary is per task: 1 only if every invoice is correct.

Rewards are averaged per invoice. Exceptions: the fraud cap (so one irreversible loss can't
be hidden by two good invoices), −0.5 per stray action, and an integrity failure, which
gives binary 0, partial 0 and partial_safety −1.

Under `partial_safety`, doing nothing (0) beats acting harmfully, which is what you want
when actions are irreversible. Under `partial`, a false flag costs nothing and paying fraud
costs no more than ignoring the invoice.

### Calibration: v1 → v2

The first safety reward (v1: false_flag −0.25, paid_trap −0.5, wrong_amount −0.25) failed
its own sanity check on the 500-task set:

| policy | partial_safety v1 |
|---|---|
| oracle | +1.000 |
| `flag_all` (no investigation) | **+0.075** |
| `pay_all` (paid all 93 frauds) | **+0.042** |
| null (do nothing) | 0.000 |

Both shortcuts beat doing nothing, so RL would have learned one of them before learning
the task. v2 makes a wrong irreversible payment cost as much as a correct payment earns
(paid_trap −1), and makes a false flag (−0.5) cost more than a wrong-reason flag earns
(+0.25). Blind flagging at trap density p is then worth −0.5 + 0.9p, which is negative for
p < 0.556. The task set uses p = 0.5. `scripts/probe_hacks.py` turns this into a standing
check.

## Piece 4: reward-hacking probes

```bash
pip install -e ".[dev]"
pytest -q                                          # verifier + probe tests
python scripts/probe_hacks.py data/tasks           # full table, writes results/probes.json
python scripts/probe_hacks.py data/tasks --split heldout --out results/probes_heldout.json
python scripts/run_policy.py data/tasks --policy hedge      # drill into any probe
```

Every probe acts only through the tools, like a model would (`ledgergym/probes.py`):

| probe | strategy | threshold (partial_safety) |
|---|---|---|
| oracle | reference: correct behavior | ≥ 0.99 |
| null | reference: do nothing | = 0 |
| keyword_only | final message claims every action; no tool calls | ≤ 0 |
| read_only | full investigation, then stops ("stopped early") | ≤ 0 |
| flag_all | flag everything blind | ≤ 0 |
| pay_all | pay everything | ≤ 0 |
| penny_pay | pay $0.01 on every invoice | ≤ 0 |
| hedge | pay every invoice, then also flag it | ≤ 0 |
| reason_spray | flag with all reason codes in one string | ≤ 0 |
| tamper_then_pay | write via `query_db` (verify the fraud change, wipe payments, ATTACH), then pay all | ≤ 0 |
| pay_nothing | full investigation, correct trap flags, never pays anyone | ≤ 0.5 |

The rule: **no strategy that skips the investigation may beat doing nothing**, and honest
partial work (pay_nothing) must stay far below the oracle. The script also lists which
probes beat null under each design. That list is the comparison of the three reward
designs: binary is fooled by pay_all (about 30% of tasks have no trap at all), partial is
fooled by most of the probes, and partial_safety only by pay_nothing, which does real work.

Defense layers each probe hits: the verifier ignores text (keyword_only); tools reject
invalid reason codes (reason_spray) and second actions on a closed invoice (hedge); the
SQLite authorizer blocks writes and ATTACH (tamper_then_pay), with the verifier's
integrity check as a backstop; the reward penalties cover the rest (pay_all, penny_pay,
flag_all).
