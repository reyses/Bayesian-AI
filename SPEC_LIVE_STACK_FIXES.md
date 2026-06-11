# SPEC: live/ Stack — Entry Point, Telemetry Indices, Recovery Fixes

**Scope:** `live/__main__.py`, `live/launcher.py`, `live/live_engine.py`, `live/engine_v2.py`, `live/diagnostic_run.py`, `live/order_manager.py`, `live/state/` artifacts
**Context:** No P0 money-loss bug. Three P1s: wrong default engine with no account whitelist, V1 feature-index relics in all live telemetry, crash recovery silently disarming exit protection. Plus a P2 batch.
**Dependencies:** Fix 2 shares the index-constant approach with SPEC_FPS_LEDGER_FIXES.md Fix 3 — if that spec's optional "hoist shared indices into core_v2/features.py" step was taken, import those constants here instead of re-deriving.
**Preserved invariants:** Ledger mutations ONLY on FILL confirmation. OrderRecord intent/state machine untouched except the BAY_CLOSE key change in Fix 4.3. MockBridge zero-slip `requested_price` path untouched.

---

## Fix 1 (P1-A) — Repoint default entry to engine_v2; archive legacy engine; harden account gate

**Problem:** `python -m live` → `launcher.py` → `live_engine.py` (V1 BlendedEngine stack), not the production `engine_v2.py` (L5). The legacy path accepts any `--account` with no sim whitelist and no confirm. `launcher._kill_stale_live_engines()` kills any python process whose cmdline contains 'live' — including a running engine_v2 session. `live_engine_legacy.py` is already archived; `live_engine.py` + `launcher.py` are the same lineage one generation newer and equally superseded.

**Changes:**
1. `live/__main__.py`: replace `from live.launcher import main` with `from live.engine_v2 import main`. `python -m live` and `python -m live.engine_v2` now both run L5.
2. Move `live/launcher.py` and `live/live_engine.py` to `live/archive/`. Pre-move grep for importers of either (`from live.launcher`, `from live.live_engine`, `live.launcher`, `live.live_engine`); expected importers are only `__main__.py` (fixed in step 1) and each other. If anything else imports them, stop and report.
3. `live/maintenance.py` prints "Run: python -m live.launcher" — update to `python -m live.engine_v2`. Also note: maintenance's `load_state()` is imported by the archived live_engine; after archival, check whether maintenance.py itself has any remaining live importer. If not, leave it (it's still a useful standalone warmup tool) but update its banner text.
4. Harden `engine_v2._resolve_account`: keep the current prompt-and-confirm for interactive runs, but add a hard refuse path matching the legacy `_ALLOWED_SIM_ACCOUNTS` intent: if the resolved account is not in `KNOWN_SIM_ACCOUNTS` AND `--i-know-this-is-a-real-account` (new explicit flag) is absent, raise SystemExit — even when passed via `--account`. The current behavior (warn but proceed on explicit `--account`) is exactly how a typo routes orders to a funded account in a headless/scripted run.
5. Do NOT port `_kill_stale_live_engines` anywhere. If stale-process cleanup is wanted later, it must match on the exact module path (`live.engine_v2`) and exclude the current process tree, with a `--kill-stale` opt-in flag.

**Acceptance:** `python -m live --headless` exits demanding `--account`; `python -m live --headless --account RealMoney123` exits with the hard-refuse message; `--account Sim101` proceeds. Grep confirms no live (non-archive) module imports launcher or live_engine.

---

## Fix 2 (P1-B) — V1 index relics in live telemetry

**Problem:** Hardcoded V1 91-D indices used against V2 201-feature vectors:
- `engine_v2.py` step 7 and `_step3_warmup`/`_step5_catchup`: `feat[12]` as z, `feat[14]` as vr, `feat[15]` as vel.
- `diagnostic_run.py`: same `feat[12]/[14]/[15]` triple, plus ledger columns.
- (`live_engine.py` uses `feat[10]/[12]` — archived by Fix 1, no patch needed.)

In V2 canonical order these land in the 5s L2 block — volume acceleration family, not 1m z_se. L5Decider itself is SAFE (reads V2 by name via `v2_getter` + `FEATURE_NAMES.index`), so no trading decision was affected. But every `v2_ledger_*.csv` z/vr/vel column, every dashboard z display and z-derived `belief_pct`, and every "WARMED UP: z=..." log line is junk. Any live-vs-backtest parity work that compared the ledger z column is invalid.

**Changes:**
1. Single source of truth: if SPEC_FPS_LEDGER_FIXES Fix 3's optional hoist was done, import from there. Otherwise define once at the top of `engine_v2.py` and import into `diagnostic_run.py`:

```python
from core_v2.features import FEATURE_NAMES
IDX_1M_Z_SE   = FEATURE_NAMES.index('L3_1m_z_se_15')
IDX_1M_VR     = FEATURE_NAMES.index('L3_1m_reversion_prob_15')   # see note
IDX_1M_VEL    = FEATURE_NAMES.index('L2_1m_price_velocity_15')
```

   **Note on "vr":** the V1 meaning of `feat[14]` was "1m variance ratio". V2 has no variance-ratio column. Decide the replacement: `L3_1m_reversion_prob_15` (closest semantic cousin for "is this mean-reverting") or `L3_1m_swing_noise_15` (chop proxy). Default to reversion_prob; rename the ledger column from `vr` to `rprob` so no one mistakes it for the old metric. Flag in the PR if the user wants swing_noise instead.
2. Replace all `feat[12]`, `feat[14]`, `feat[15]` reads in engine_v2.py (status line, ledger writes, GUI pushes, warmup verification logs, `z_pct` computation) and diagnostic_run.py (logs + ledger columns) with the named constants.
3. `grep -rn "feat\[1[0-9]\]\|\[12\]\|\[14\]\|\[15\]" live/ --include="*.py"` (excluding archive/) returns nothing after the change.
4. **Contamination note:** add `reports/findings/<date>_live_ledger_z_contamination.md` listing the affected files: all `reports/live/v2_ledger_*.csv` and `reports/live/diagnostic_ledger.csv` produced before this fix — their z/vr/vel columns are 5s vol-accel values, not 1m z. Trade decisions and PnL columns in the same files remain valid (decisions came from L5/v2_getter, fills from NT8).

**Acceptance:** One mock run produces a ledger whose z column matches `L3_1m_z_se_15` from the same bars' V2 features (spot-check 5 rows). Dashboard z display moves plausibly in ±3 range instead of vol-accel scale.

---

## Fix 3 (P1-C) — Recovery restores positions with disarmed exit protection

**Problem:** `_step5b_recover_trade` restores direction/entry_price/entry_tier via `add_position`, but `peak_pnl` restarts at 0 (the checkpoint saves it; add_position can't accept it). A trade that was +$80 at crash comes back with giveback/MFE-armed exits disarmed until it re-peaks — the exact protection a recovered position most needs. Secondary: direction check validates only the primary; chain count is never reconciled against NT8 qty.

**Changes:**
1. `core_v2/ledger.py`: add optional `restore_peak_pnl: float = 0.0` and `restore_extras: Optional[dict] = None` params to `add_position`, applied after Position construction. Document: "recovery-only; never set on fresh entries."
2. `engine_v2._step5b_recover_trade`: pass `restore_peak_pnl=sp.get('peak_pnl', 0.0)`. The checkpoint already serializes peak_pnl per position (`_periodic_save` writes it) — it was saved and then dropped on the floor at restore.
3. Qty reconciliation: after restoring, compare `self._pos_ledger.n_contracts` to `abs(self._orders.nt8_qty)`. On mismatch, log ERROR with both numbers and fall back to monitoring-only (do not restore — clear the ledger) rather than managing a position whose size we have wrong. NT8 is ground truth.
4. Chain direction: the existing primary-direction check is sufficient given the ledger enforces chain direction == primary at add time; no extra check needed — but the add loop must restore the primary FIRST (it already does, positions[0]) and wrap chain adds in try/except so one bad chain row doesn't crash startup; on exception, log and fall back to monitoring-only.
5. Extras caveat: `position.extras['thresholds']` (adaptive SL/TP) is not serialized. After restore, exits run on constructor defaults. Acceptable — note it in the recovery log line: `(thresholds=defaults, peak restored=$X)`.

**Acceptance:** Unit-style test (mock checkpoint dict + fake OrderManager shadow): position saved with peak_pnl=80 restores with `ledger.primary.peak_pnl == 80`; mismatch nt8_qty=2 vs saved 1 position → ledger stays flat, error logged.

---

## Fix 4 (P2 batch)

1. **Step-4 gap check estimate.** `oldest_dump_ts = last_ts - bar_count*5` assumes contiguous 5s bars; weekend/maintenance gaps inside the dump shrink the estimate and can mask a real ATLAS↔dump hole. Track the actual first dump bar: in the Step-4 loop, on the first BAR message record `first_dump_ts = bar['timestamp']`; use it directly in the gap computation. Delete the estimation comment block.
2. **Magic date in `_step2_build`.** `f >= '2026_03_20'` hardcodes the dataset epoch. Move to a module constant `FEATURES_EPOCH = '2026_03_20'` with a comment ("first day with valid ATLAS_NT8 5s data; earlier days lack ..."), and log when days are excluded by it. Constant, not config — but named and discoverable.
3. **BAY_CLOSE OrderRecord overwrite.** `self._orders['BAY_CLOSE'] = rec` clobbers the previous close's record each time, destroying handshake audit history. Keep the wire order_id 'BAY_CLOSE' (bridge contract) but store under a unique key: `self._orders[f'BAY_CLOSE#{self._close_seq}']` with `self._close_seq += 1`, and keep a `self._orders['BAY_CLOSE']` alias pointing at the latest so `on_fill`/`on_order_ack` lookups by wire id still resolve. Lowest-risk implementation: a small `_resolve_rec(oid)` helper that maps 'BAY_CLOSE' → latest seq key; all message handlers go through it.
4. **B9 K=5 clock skew (doc-only).** Live `_PosTraj.entry_bar_count` snapshots when the ledger first SEES the position (post-FILL), so B9 fires ~1 bar later than the backtest's entry-bar anchor. Magnitude: one 5s bar. Add a note to the parity doc (docs/Active/LIVE_L5_ARCHITECTURE.md or nearest equivalent) so a future 1-bar B9 timing discrepancy isn't chased as a bug.
5. **Stale legacy state file.** `live/state/active_trade.json` belongs to the archived legacy engine's recovery flow. Delete it in the same commit as Fix 1 (its restore logic lives only in the archived file). Leave `checkpoint.json` (v5, engine_v2's) untouched.
6. **maintenance.py banner** — covered in Fix 1.3.

---

## Execution order & gates

1. **Fix 2** first (telemetry indices) — isolated, makes every subsequent test run produce trustworthy ledgers.
2. **Fix 1** (entry point + archive + account gate). Single commit for the archive moves + __main__ repoint + active_trade.json deletion (Fix 4.5).
3. **Fix 3** (recovery) — touches core_v2/ledger.py; coordinate with any in-flight work from SPEC_FPS_LEDGER_FIXES on the same file.
4. **Fix 4.1–4.3** batch.

**Global gate:** one full MockBridge replay (`python -m live.engine_v2 --mock --mock-day <reference day>`) must complete end-to-end after each numbered fix: HISTORY_DONE → live replay → MOCK_DONE → clean shutdown, with the v2_ledger z column passing the Fix-2 spot check and at least one ENTRY/EXIT round trip reconciling (no RECONCILE MISMATCH lines). Archive the pre-fix and post-fix mock ledgers side by side in `reports/findings/` for the contamination note.
