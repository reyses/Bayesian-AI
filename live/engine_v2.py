"""
Live Engine V2 — production pipeline, 7-step startup, SIM account.

1. CHECK    → Is ATLAS_NT8 current? If not, dump missing days
2. BUILD    → Build features for any new days
3. WARMUP   → Load ATLAS_NT8 + ATLAS_LIVE delta into aggregator
4. SYNC     → Connect NT8, receive history bars
5. CATCH-UP → Compute features until Python time == NT8 time
6. VERIFY   → Latency < 1s? Sync confirmed?
7. TRADE    → Engine makes decisions, orders via OrderManager

Runs the L5 zigzag engine (L5Decider): zigzag/R-trigger entries with the
B7/B9/B10 sizing stack on V2 streaming features.
All orders go to NT8 SIM account. No dry-run — SIM IS the test.

Outputs:
    reports/live/v2_ledger_YYYY_MM_DD.csv   — every 5s bar + features + state
    reports/live/v2_trades_YYYY_MM_DD.csv   — entry/exit events for parity check

Usage:
    python -m live.engine_v2                       # full run (prompts for account)
    python -m live.engine_v2 --account Playback101 # NT8 Market Replay (playback)
    python -m live.engine_v2 --skip-check          # skip step 1 (assume current)
    python -m live.engine_v2 --skip-build          # skip step 2
    python -m live.engine_v2 --headless            # no dashboard GUI
"""

import warnings

warnings.filterwarnings("ignore", module="numba")

import asyncio
import argparse
import json
import logging
import os
import sys
import time
import glob
import numpy as np
import pandas as pd

class DummyModel:
    def predict(self, df):
        return [0.0] * len(df)
    def predict_proba(self, df):
        return [[0.5, 0.5]] * len(df)

from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("engine_v2")

from live.config import LiveConfig
from live.nt8_client import NT8Client
from live.protocol import MsgType, subscribe, place_order, close_position
from core_v2.features import FEATURE_NAMES, N_FEATURES

IDX_1M_Z_SE = FEATURE_NAMES.index("L3_1m_z_se_15")
IDX_1M_RPROB = FEATURE_NAMES.index("L3_1m_reversion_prob_15")
IDX_1M_VEL = FEATURE_NAMES.index("L2_1m_price_velocity_15")
FEATURES_EPOCH = (
    "2026_03_20"  # first day with valid ATLAS_NT8 5s data; earlier days lack...
)

from config.symbols import SYMBOL_MAP
from live.order_manager import OrderManager
from live.gui_bridge import GUIBridge
from core_v2.ledger import Ledger
from core_v2.sim_executor import (
    _compute_entry_context,
    force_close as ledger_force_close,
)

TICK = 0.25
TV = 0.50

# Paths
ATLAS_NT8 = "DATA/ATLAS_NT8"
ATLAS_LIVE = "DATA/ATLAS_LIVE"
# Features live INSIDE each atlas folder now
FEATURES_LIVE = "DATA/ATLAS_LIVE/FEATURES_5s"
FEATURES_NT8 = "DATA/ATLAS_NT8/FEATURES_5s"
NT8_CONFIG = "config/nt8_dataset.json"

# Checkpoints — dual system (NT8 seed + live rolling)
NT8_CHECKPOINT = os.path.join(ATLAS_NT8, "checkpoint.json")
LIVE_CHECKPOINT = "live/state/checkpoint.json"

# Sync thresholds
MAX_SYNC_LAG_S = 10.0  # hard ceiling — lag > this blocks trading
BAR_PERIOD_S = 5  # 5s primary bar; NT8 sends OPEN time, so true bar_age = wall - (ts + BAR_PERIOD_S)
HONING_TARGET_LAG_S = 3.0  # target bar_age (post-close) during step 6 honing loop
HONING_TIMEOUT_S = 30.0  # max time to spend honing before proceeding with warning
WARMUP_DAYS = 5  # days of history to load for aggregator context

# Shutdown
CUDA_CLOSE_TIMEOUT_S = 5.0  # max time to wait for cuda.close() during shutdown


class LiveEngineV2:
    """Production live engine — 7-step startup, physics-only."""

    def __init__(
        self,
        config: LiveConfig,
        skip_check: bool = False,
        skip_build: bool = False,
        gui_queue=None,
        shared_state=None,
        mock_client=None,
    ):
        self._cfg = config
        self._skip_check = skip_check
        self._skip_build = skip_build
        self._mock_client = mock_client  # MockBridge instance, or None for real NT8
        self._shared_state = shared_state or {}

        self._asset = SYMBOL_MAP.get(config.asset_ticker)
        if self._asset is None:
            raise ValueError(f"Unknown asset: {config.asset_ticker}")

        # Core components (initialized in startup steps)
        self._client = None
        self._lfe = None  # LiveFeatureEngineV2 (V2-only streaming features)
        self._engine = None

        # Position ledger — single source of truth for engine's position view.
        # Mutated only on confirmed fills (via on_fill → ledger.add/remove_position).
        self._pos_ledger = Ledger()

        # Order management (NT8 handshake + in-flight tracking)
        self._orders = OrderManager(config, self._pos_ledger)

        # Dashboard
        self._gui = GUIBridge(gui_queue)

        # State
        self._bar_count = 0
        self._feat_count = 0
        self._synced = False
        self._trading = False
        self._shutting_down = False
        self._broker_connected = True  # NT8 <-> broker status (assume OK at start)
        self._nt8_realized_pnl = None  # set from ACCOUNT_UPDATE messages
        self._nt8_unrealized_pnl = 0.0
        self._nt8_cash_value = 0.0
        self._daily_pnl = 0.0
        self._trade_count = 0
        self._last_ts = 0.0
        self._last_price = 0.0
        # Default to wall-clock date; mock mode overrides to the replay day
        # so multi-day mock runs don't clobber each other's output files.
        self._session_date = time.strftime("%Y_%m_%d")
        if mock_client is not None and hasattr(mock_client, "_day_to_replay"):
            mock_day = getattr(mock_client, "_day_to_replay", None)
            if mock_day:
                self._session_date = mock_day

        # Ledger (every 5s bar) + trade log (entry/exit events)
        self._ledger = None
        self._ledger_path = None
        self._trade_log = None
        self._trade_log_path = None
        # NT8 ground-truth trade log (from TRADE_CLOSED events)
        self._nt8_trade_log = None
        self._nt8_trade_log_path = None

        # Live bar capture
        self._live_bars = []
        self._live_79d = []

    # ═══════════════════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════════════════

    async def run(self):
        """Execute the 7-step startup then trade."""
        logger.info("=" * 60)
        logger.info("LIVE ENGINE V2 — Physics Only")
        logger.info(f"  Instrument: {self._cfg.instrument}")
        logger.info(f"  Account:    {self._cfg.account}")
        logger.info(f"  Account:    {self._cfg.account} (SIM)")
        logger.info("=" * 60)

        try:
            # Step 0: clean live accumulation folders (fresh start every session)
            self._step0_clean_live()

            # Steps 1-2: offline (no connection needed)
            if not self._skip_check:
                self._step1_check()
            if not self._skip_build:
                self._step2_build()

            # Step 3: warmup from disk
            self._step3_warmup()

            # Steps 4-7: online (need NT8 connection or mock)
            if self._mock_client:
                self._client = self._mock_client
                # Mock mode: replay full day, ignore checkpoint's last_ts
                self._last_ts = 0
                logger.info("  Using MockBridge (replay mode)")
            else:
                self._client = NT8Client(self._cfg)
            # Tell NT8/mock to only send bars after our warmup
            if self._last_ts > 0:
                self._client.set_resume_timestamp(self._last_ts)
                logger.info(f"  Delta sync from {self._ts_str(self._last_ts)}")
            connected = await self._client.connect()
            if not connected:
                logger.error("Failed to connect to NT8")
                return

            await self._step4_sync()
            if self._shutting_down:
                return
            await self._step5_catchup()
            if self._shutting_down:
                return
            self._step5b_recover_trade()
            if self._mock_client:
                # Mock mode: skip honing, go straight to trading
                self._synced = True
            else:
                await self._step6_verify()
                if self._shutting_down:
                    return

            if self._synced:
                await self._step7_trade()
            else:
                logger.error("Sync failed — not trading")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal: {e}", exc_info=True)
        finally:
            try:
                await self._shutdown()
            except Exception as e:
                logger.error(f"Shutdown failed: {e}")
                # Even if shutdown fails, force-exit so the process doesn't hang
                os._exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 0: CLEAN — fresh start for live accumulation
    # ═══════════════════════════════════════════════════════════════════

    def _step0_clean_live(self):
        """Clear live accumulation folders so each session starts clean.

        Previous session data is already in ATLAS_NT8 (via the converter).
        The live folders are a rolling buffer — stale fragments from prior
        sessions cause patchy data and confuse warmup.
        """
        logger.info("")
        logger.info("STEP 0: CLEAN LIVE FOLDERS")

        import shutil

        cleaned = 0

        # Clear live feature parquets
        if os.path.exists(FEATURES_LIVE):
            for f in glob.glob(os.path.join(FEATURES_LIVE, "*.parquet")):
                os.remove(f)
                cleaned += 1

        # Clear live bar chunks
        chunks_dir = os.path.join(ATLAS_LIVE, "5s", "_chunks")
        if os.path.exists(chunks_dir):
            for f in glob.glob(os.path.join(chunks_dir, "*.parquet")):
                os.remove(f)
                cleaned += 1

        # Reset in-memory accumulators
        self._live_bars.clear()
        self._live_79d.clear()

        logger.info(f"  Cleaned {cleaned} stale files")
        logger.info(f"  {FEATURES_LIVE}/ — empty")
        logger.info(f"  {ATLAS_LIVE}/5s/_chunks/ — empty")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: CHECK — is ATLAS_NT8 current?
    # ═══════════════════════════════════════════════════════════════════

    def _step1_check(self):
        logger.info("")
        logger.info("STEP 1: CHECK ATLAS_NT8")

        atlas_5s = os.path.join(ATLAS_NT8, "5s")
        if not os.path.exists(atlas_5s):
            logger.warning(f"  {atlas_5s}/ not found — run history dump first")
            return

        files = sorted(f for f in os.listdir(atlas_5s) if f.endswith(".parquet"))
        if not files:
            logger.warning("  No parquets in ATLAS_NT8/5s/")
            return

        last_day = files[-1].replace(".parquet", "")
        today = time.strftime("%Y_%m_%d")
        logger.info(f"  Last NT8 day: {last_day}")
        logger.info(f"  Today:        {today}")

        if last_day >= today:
            logger.info("  ATLAS_NT8 is current")
        else:
            logger.warning(f"  ATLAS_NT8 is {last_day}, today is {today}")
            logger.warning(f"  Run: python tools/convert_nt8_atlas.py")
            logger.warning(f"  Or use diagnostic_run.py to dump from NT8")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: BUILD — features for new days
    # ═══════════════════════════════════════════════════════════════════

    def _step2_build(self):
        logger.info("")
        logger.info("STEP 2: BUILD FEATURES")

        # Refresh the B10 data chain so the day-regime sizer has a
        # current row for today. Two steps, both unconditional (run
        # before the "features up to date" early-return) and both
        # NON-FATAL (failure -> B10 degrades, never crashes):
        #
        #   2a. fetch_vix_dxy.py        -- network (yfinance), timeout 90s.
        #       On failure, the existing raw VIX/DXY parquets are reused
        #       (slightly stale; only affects low-importance B10 features).
        #   2b. build_cross_day_features.py -- offline, ~10-30s. Reads the
        #       (possibly just-refreshed) raw parquets + the 1m bars Step 1
        #       refreshed. On failure, B10 falls back to NORMAL mode.
        import subprocess as _sp

        logger.info("  L5: refreshing VIX/DXY (yfinance)...")
        try:
            vd = _sp.run(
                [sys.executable, "tools/sourcing/fetch_vix_dxy.py"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            if vd.returncode == 0:
                logger.info("  VIX/DXY refreshed")
            else:
                logger.warning(
                    f"  VIX/DXY fetch failed (using existing "
                    f"raw parquets): {vd.stderr[-160:]}"
                )
        except Exception as e:
            logger.warning(
                f"  VIX/DXY fetch skipped (network?) (using existing raw parquets): {e}"
            )

        logger.info("  L5: rebuilding cross_day_features for B10...")
        try:
            cd_result = _sp.run(
                [sys.executable, "tools/sourcing/build_cross_day_features.py"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if cd_result.returncode == 0:
                logger.info("  cross_day_features rebuilt (B10 ready)")
            else:
                logger.warning(
                    f"  cross_day_features build failed "
                    f"(B10 -> NORMAL mode): "
                    f"{cd_result.stderr[-200:]}"
                )
        except Exception as e:
            logger.warning(
                f"  cross_day_features build error (B10 -> NORMAL mode): {e}"
            )

        os.makedirs(FEATURES_NT8, exist_ok=True)
        atlas_5s = os.path.join(ATLAS_NT8, "5s")
        if not os.path.exists(atlas_5s):
            return

        existing = {
            f.replace(".parquet", "")
            for f in os.listdir(FEATURES_NT8)
            if f.endswith(".parquet")
        }
        atlas_days = set()
        excluded_days = 0
        for f in os.listdir(atlas_5s):
            if f.endswith(".parquet"):
                day_str = f.replace(".parquet", "")
                if day_str >= FEATURES_EPOCH:
                    atlas_days.add(day_str)
                else:
                    excluded_days += 1
        if excluded_days > 0:
            logger.info(
                f"  Excluded {excluded_days} early days before epoch ({FEATURES_EPOCH})"
            )
        missing = sorted(atlas_days - existing)

        if not missing:
            logger.info(f"  Features up to date ({len(existing)} days)")
            return

        logger.info(f"  Missing: {len(missing)} days ({missing[0]} to {missing[-1]})")
        logger.info(f"  Building...")

        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "core_v2/build_dataset.py",
                "--atlas",
                ATLAS_NT8,
                "--start",
                missing[0].replace("_", "-"),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode == 0:
            logger.info(f"  Features built")
        else:
            logger.warning(f"  Build failed: {result.stderr[-200:]}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: WARMUP — load history into aggregator
    # ═══════════════════════════════════════════════════════════════════

    def _step3_warmup(self):
        logger.info("")
        logger.info("STEP 3: WARMUP")

        # LiveFeatureEngine: Clean buffer guaranteeing 100% SFE offline parity
        from core_v2.live_features import LiveFeatureEngine

        self._lfe = LiveFeatureEngine(ATLAS_NT8, v2_only=True)
        logger.info("  Feature engine: L5 (V2-ONLY -- V1 SFE purged)")
        # Mock mode: skip the day being replayed so its bars arrive via on_bar
        exclude_day = None
        if self._mock_client and hasattr(self._mock_client, "_day_to_replay"):
            exclude_day = self._mock_client._day_to_replay
            logger.info(
                f"  Mock mode: excluding {exclude_day} from warmup (will replay)"
            )
        bar_counts = self._lfe.load_history(exclude_day=exclude_day)
        # Pretty-print bar counts per TF on one line for diagnostics
        bc_str = "  ".join(f"{tf}={n:,}" for tf, n in sorted(bar_counts.items()))
        logger.info(f"  Loaded: {bc_str}")

        # Show last bar timestamp per TF (helps spot stale data)
        if "5s" in self._lfe._bars and len(self._lfe._bars["5s"]) > 0:
            last_5s = float(self._lfe._bars["5s"]["timestamp"].iloc[-1])
            age_s = time.time() - last_5s
            logger.info(
                f"  Last 5s bar: {self._ts_str(last_5s)} ({age_s / 60:.0f} min ago)"
            )

        # Load velocities from checkpoint (newest of LIVE vs NT8)
        best_path, best_ts = None, 0
        for path in [LIVE_CHECKPOINT, NT8_CHECKPOINT]:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        cp = json.load(f)
                    ts = cp.get("last_ts", 0)
                    if ts > best_ts:
                        best_path, best_ts = path, ts
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"  Corrupt checkpoint: {path}")

        self._saved_ledger_state = {}
        if best_path:
            with open(best_path, encoding="utf-8") as f:
                cp = json.load(f)
            # Mock mode: skip velocities AND accumulators — checkpoint has
            # state from a future session. Mock replays the past with fresh
            # aggregation (same as build_dataset does).
            if self._mock_client:
                logger.info("  Mock mode: velocities + accumulators reset (fresh)")
            else:
                self._lfe.load_velocities(cp.get("velocities", {}))
                if cp.get("accumulators"):
                    self._lfe.load_accumulators(cp["accumulators"])
                    logger.info(
                        f"  Accumulators restored ({len(cp['accumulators'])} TFs)"
                    )
            # Support both v3 (trade_state) and v4 (ledger_state) checkpoints
            if "ledger_state" in cp:
                self._saved_ledger_state = cp["ledger_state"]
            elif "trade_state" in cp:
                # Convert v3 trade_state to v4 ledger_state for backward compat
                ts = cp["trade_state"]
                positions = []
                if ts.get("in_pos", False):
                    positions.append(
                        {
                            "direction": ts.get("direction", ""),
                            "entry_price": ts.get("entry_price", 0),
                            "entry_ts": ts.get("entry_ts", 0),
                            "entry_tier": ts.get("entry_tier", "UNKNOWN"),
                            "is_chain": False,
                        }
                    )
                    for cc in ts.get("chains", []):
                        positions.append(
                            {
                                "direction": cc.get("direction", ""),
                                "entry_price": cc.get("entry_price", 0),
                                "entry_ts": cc.get("entry_ts", 0),
                                "entry_tier": cc.get("entry_tier", "UNKNOWN"),
                                "is_chain": True,
                            }
                        )
                self._saved_ledger_state = {"positions": positions}
            self._last_ts = cp.get("last_ts", 0)
            cp_age_min = (time.time() - cp.get("last_ts", 0)) / 60
            logger.info(
                f"  Checkpoint: {os.path.basename(best_path)} "
                f"(age {cp_age_min:.0f} min, {len(cp.get('velocities', {}))} velocities)"
            )
        else:
            logger.warning("  No checkpoint found — cold start")

        from live.l5_decider import L5Decider, L5Context

        pivot_source = getattr(self._cfg, "pivot_source", "stream")
        l5_ctx = L5Context.load(pivot_source=pivot_source)
        self._engine = L5Decider(l5_ctx)
        logger.info(f"  Engine: L5Decider (pivot_source={pivot_source})")
        if pivot_source == "stream":
            if "1m" in self._lfe._bars and len(self._lfe._bars["1m"]) > 0:
                self._engine.prime_atr_from_history(self._lfe._bars["1m"])

        # Save pre-loaded end timestamp for gap check in Step 4
        self._preloaded_end_ts = self._last_ts

        # Verify: compute one feature from last loaded bar
        if "1m" in self._lfe._bars and len(self._lfe._bars["1m"]) > 0:
            last_1m_ts = float(self._lfe._bars["1m"]["timestamp"].iloc[-1])
            feat = self._lfe._compute_features(last_1m_ts)
            if feat is not None:
                logger.info(
                    f"  WARMED UP: z={feat[IDX_1M_Z_SE]:.2f} rprob={feat[IDX_1M_RPROB]:.2f} "
                    f"| Orders={len(self._orders._orders)} "
                    f"1h={len(self._lfe._bars.get('1h', []))}"
                )
            else:
                logger.warning("  Warmup feature returned None")
        else:
            logger.warning("  No 1m bars loaded")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: SYNC — connect NT8, receive history
    # ═══════════════════════════════════════════════════════════════════

    async def _step4_sync(self):
        logger.info("")
        logger.info("STEP 4: SYNC NT8")

        # Receive history dump
        history_done = False
        bar_count = 0
        first_dump_ts = 0
        t0 = time.perf_counter()

        while not history_done:
            if self._shared_state.get("shutdown"):
                logger.info("  Step 4 aborted by shutdown request")
                self._shutting_down = True
                return
            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=2.0)
            except asyncio.TimeoutError:
                # Bounded wait so we can re-check shutdown flag — overall hard
                # cap is 120s of history idle (60 iterations of 2s).
                if (time.perf_counter() - t0) > 120.0:
                    logger.warning("  History timeout (120s)")
                    break
                continue

            if msg.get("type") == MsgType.BAR:
                bar = self._extract_bar(msg)
                if first_dump_ts == 0:
                    first_dump_ts = bar["timestamp"]
                self._ingest_bar(bar)
                bar_count += 1
            elif msg.get("type") == MsgType.HISTORY_DONE:
                history_done = True
            elif msg.get("type") == "CONNECTED":
                nt8_acct = msg.get("account")
                logger.info(
                    f"  NT8: account={nt8_acct} instrument={msg.get('instrument')}"
                )
                if nt8_acct and nt8_acct != self._cfg.account:
                    logger.warning(
                        f"  ACCOUNT MISMATCH: engine configured for "
                        f'"{self._cfg.account}" but the NT8 bridge reports '
                        f'"{nt8_acct}". Orders may route to the wrong '
                        f"account — fix --account or the NT8 strategy "
                        f"account before trading."
                    )

        elapsed = time.perf_counter() - t0
        logger.info(f"  History: {bar_count:,} bars in {elapsed:.1f}s")
        logger.info(f"  Bars: {self._lfe.bar_counts}")

        # ── Gap check: is there a hole between ATLAS_NT8 and the dump? ──
        # Skip in mock mode — mock replays ATLAS bars, no gap possible.
        # The pre-loaded ATLAS_NT8 data has a latest timestamp. The oldest
        # bar from the NT8 dump should be within 5 minutes of that. If not,
        # there's a gap of missing bars that produces stale features.
        if self._mock_client:
            return
        MAX_GAP_S = 300  # 5 minutes
        if "5s" in self._lfe._bars and len(self._lfe._bars["5s"]) > 0:
            # _last_ts before Step 4 = end of pre-loaded ATLAS_NT8 data
            # _last_ts after Step 4 = end of NT8 dump (latest bar received)
            # The pre-loaded end was saved before we overwrote _last_ts
            preloaded_end = getattr(self, "_preloaded_end_ts", 0)
            if preloaded_end > 0 and bar_count > 0:
                # We tracked first_dump_ts when the first bar arrived.
                gap_s = first_dump_ts - preloaded_end
                if gap_s > MAX_GAP_S:
                    gap_min = gap_s / 60
                    logger.error(
                        f"  GAP DETECTED: {gap_min:.0f} min hole between ATLAS_NT8 and NT8 dump"
                    )
                    logger.error(
                        f"  ATLAS_NT8 ends at {self._ts_str(preloaded_end)}, "
                        f"dump starts {self._ts_str(first_dump_ts)}"
                    )
                    logger.error(
                        f"  Run: python tools/convert_nt8_atlas.py --contract MNQ_06-26"
                    )
                    logger.error(
                        f"  Then: python core_v2/build_dataset.py --resolution 5s --atlas DATA/ATLAS_NT8 --start <date>"
                    )
                    logger.error(f"  REFUSING TO TRADE — features would be stale.")
                    self._shutting_down = True
                    return

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: CATCH-UP — process bars until current
    # ═══════════════════════════════════════════════════════════════════

    async def _step5_catchup(self):
        logger.info("")
        logger.info("STEP 5: CATCH-UP")

        # Feed bars into LiveFeatureEngineV2 until caught up to wall time.
        catchup_bars = 0
        wall_time = time.time()

        while True:
            if self._shared_state.get("shutdown"):
                logger.info("  Step 5 aborted by shutdown request")
                self._shutting_down = True
                return
            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=2.0)
            except asyncio.TimeoutError:
                break  # queue drained

            if msg.get("type") != MsgType.BAR:
                continue

            bar = self._extract_bar(msg)
            self._ingest_bar(bar)
            catchup_bars += 1

            # Break once we're within 10s of wall time (caught up)
            if self._last_ts > 0 and (wall_time - self._last_ts) < MAX_SYNC_LAG_S:
                break

        # Verify with one feature
        feat = self._lfe._compute_features(self._last_ts) if self._last_ts > 0 else None
        if feat is not None:
            logger.info(
                f"  Caught up: {catchup_bars:,} bars | "
                f"z={feat[IDX_1M_Z_SE]:.2f} rprob={feat[IDX_1M_RPROB]:.2f}"
            )
        else:
            logger.info(f"  Caught up: {catchup_bars:,} bars")
        logger.info(
            f"  Last bar: {self._ts_str(self._last_ts)} price={self._last_price:.2f}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5b: RECOVER — restore in-flight trade if NT8 has open position
    # ═══════════════════════════════════════════════════════════════════

    def _step5b_recover_trade(self):
        """If NT8 has an open position and checkpoint has matching ledger state,
        restore ledger positions to continue managing the trade.
        """
        nt8_qty = self._orders.nt8_qty
        saved_ledger = self._saved_ledger_state
        saved_positions = saved_ledger.get("positions", [])
        has_saved = len(saved_positions) > 0

        if nt8_qty == 0 and not has_saved:
            logger.info("  RECOVERY: both flat — clean start")
            return

        if nt8_qty == 0 and has_saved:
            logger.warning(
                "  RECOVERY: checkpoint had trade but NT8 is flat — trade closed while offline"
            )
            return

        if nt8_qty != 0 and not has_saved:
            nt8_side = self._orders.nt8_side
            logger.warning(
                f"  RECOVERY: NT8 has {nt8_side} x{nt8_qty} "
                f"but no trade in checkpoint — unknown trade, monitoring only"
            )
            return

        # Both have a position — verify direction match
        saved_dir = saved_positions[0].get("direction", "")
        nt8_dir = "long" if self._orders.nt8_side == "LONG" else "short"

        if saved_dir != nt8_dir:
            logger.error(
                f"  RECOVERY: direction mismatch — NT8={nt8_dir} "
                f"checkpoint={saved_dir} — NOT restoring"
            )
            return

        if len(saved_positions) != abs(nt8_qty):
            logger.warning(
                f"  RECOVERY: quantity mismatch — NT8={abs(nt8_qty)} "
                f"checkpoint={len(saved_positions)} — NOT restoring, monitoring only"
            )
            return

        # Restore positions into the ledger
        try:
            for sp in saved_positions:
                entry_feat = np.zeros(N_FEATURES, dtype=np.float32)
                self._pos_ledger.add_position(
                    direction=sp["direction"],
                    entry_price=sp["entry_price"],
                    entry_ts=sp.get("entry_ts", 0),
                    entry_tier=sp.get("entry_tier", "UNKNOWN"),
                    entry_features=entry_feat,
                    is_chain=sp.get("is_chain", False),
                    cnn_flipped=sp.get("cnn_flipped", False),
                    entry_abs_z=sp.get("entry_abs_z", 0),
                    entry_velocity=sp.get("entry_velocity", 0),
                    entry_h1_z=sp.get("entry_h1_z", 0),
                    entry_vol_rel=sp.get("entry_vol_rel", 0),
                    ride_exit_bars=sp.get("ride_exit_bars", 2),
                    restore_peak_pnl=sp.get("peak_pnl", 0),
                    restore_extras=sp.get("extras", None),
                )
        except Exception as e:
            logger.error(
                f"  RECOVERY: failed to restore positions ({e}) — "
                f"clearing ledger to monitor only"
            )
            self._pos_ledger._primary = None
            self._pos_ledger._chains.clear()
            self._pos_ledger._by_id.clear()
            return

        n_chains = sum(1 for sp in saved_positions if sp.get("is_chain"))
        prim = saved_positions[0]
        logger.info(
            f"  RECOVERY: restored {saved_dir} {prim.get('entry_tier')} "
            f"@ {prim.get('entry_price', 0):.2f} "
            f"(chains={n_chains}, thresholds=defaults, peak restored=${prim.get('peak_pnl', 0):.0f})"
        )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 6: VERIFY — honing loop, drain inbound until bar_age converges
    # ═══════════════════════════════════════════════════════════════════

    async def _step6_verify(self):
        """Hone in on real-time by pulling live bars until bar_age converges.

        Bar timestamps are OPEN times (NT8 OnBarClose sends Times[idx][1]), so
        true staleness is `wall - (last_ts + BAR_PERIOD_S)`. We loop pulling
        fresh bars off the inbound queue until that drops under the target, or
        until HONING_TIMEOUT_S elapses.
        """
        logger.info("")
        logger.info("STEP 6: VERIFY SYNC (honing)")

        def _bar_age(wall, ts):
            return wall - (ts + BAR_PERIOD_S) if ts > 0 else 999.0

        t0 = time.perf_counter()
        bars_honed = 0
        last_log = 0.0
        bar_age = _bar_age(time.time(), self._last_ts)

        logger.info(
            f"  start: wall={self._ts_str(time.time())} "
            f"last_bar={self._ts_str(self._last_ts)} "
            f"bar_age={bar_age:.1f}s"
        )

        while True:
            if self._shared_state.get("shutdown"):
                logger.info("  Honing aborted by shutdown request")
                self._shutting_down = True
                return
            elapsed = time.perf_counter() - t0
            bar_age = _bar_age(time.time(), self._last_ts)

            if bar_age <= HONING_TARGET_LAG_S:
                wall_time = time.time()
                logger.info(f"  Wall time:   {self._ts_str(wall_time)}")
                logger.info(
                    f"  Last bar:    {self._ts_str(self._last_ts)} "
                    f"(closed {self._ts_str(self._last_ts + BAR_PERIOD_S)})"
                )
                logger.info(f"  Bar age:     {bar_age:.1f}s (post-close)")
                logger.info(f"  Bars honed:  {bars_honed}")
                logger.info(f"  SYNC HONED — caught up to real-time")
                self._synced = True
                return

            if elapsed >= HONING_TIMEOUT_S:
                self._synced = True
                logger.warning(
                    f"  HONING TIMEOUT: bar_age={bar_age:.1f}s after "
                    f"{elapsed:.0f}s — proceeding, will enter live "
                    f"when next fresh bar arrives"
                )
                return

            # Pull next message; short timeout so we re-check bar_age frequently
            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No new bar in last 1s — log progress every 2s and loop
                if (time.perf_counter() - last_log) >= 2.0:
                    logger.info(
                        f"  honing: bar_age={bar_age:.1f}s elapsed={elapsed:.0f}s"
                    )
                    last_log = time.perf_counter()
                continue

            if msg.get("type") != MsgType.BAR:
                continue

            bar = self._extract_bar(msg)
            self._ingest_bar(bar)
            if bar["timestamp"] > self._last_ts - 1:
                bars_honed += 1

    # ═══════════════════════════════════════════════════════════════════
    # STEP 7: TRADE — main loop
    # ═══════════════════════════════════════════════════════════════════

    async def _step7_trade(self):
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 7: TRADING")
        logger.info("=" * 60)

        # Open ledger (every 5s bar) + trade log (entry/exit events only)
        os.makedirs("reports/live", exist_ok=True)
        self._ledger_path = f"reports/live/v2_ledger_{self._session_date}.csv"
        self._ledger = open(self._ledger_path, "w", encoding="utf-8")
        ledger_cols = [
            "timestamp",
            "price",
            "z",
            "rprob",
            "velocity",
            "in_pos",
            "direction",
            "tier",
            "bars_held",
            "pnl",
            "peak",
            "contracts",
            "event",
        ]
        self._ledger.write(",".join(ledger_cols) + "\n")

        # Trade log — engine's internal view (for parity vs backtest)
        self._trade_log_path = f"reports/live/v2_trades_{self._session_date}.csv"
        self._trade_log = open(self._trade_log_path, "w", encoding="utf-8")
        trade_cols = [
            "timestamp",
            "type",
            "tier",
            "direction",
            "requested_price",
            "fill_price",
            "slippage",
            "pnl",
            "bars_held",
            "exit_reason",
            "is_chain",
            "contracts",
            "daily_pnl",
        ]
        self._trade_log.write(",".join(trade_cols) + "\n")
        self._pending_requests = {}

        # NT8 trade log — ground truth from TRADE_CLOSED events (for reconciliation)
        self._nt8_trade_log_path = f"reports/live/nt8_trades_{self._session_date}.csv"
        self._nt8_trade_log = open(self._nt8_trade_log_path, "w", encoding="utf-8")
        nt8_cols = [
            "fill_time",
            "order_id",
            "side",
            "entry_price",
            "exit_price",
            "pnl",
            "qty",
            "is_chain",
        ]
        self._nt8_trade_log.write(",".join(nt8_cols) + "\n")

        self._trading = True

        logger.info(f"  Ledger: {self._ledger_path}")
        logger.info(f"  Trades: {self._trade_log_path}")

        # Signal mock bridge that Step 7 is ready for live bars
        if self._mock_client and hasattr(self._mock_client, "signal_live_ready"):
            self._mock_client.signal_live_ready()

        logger.info(f"  Listening for bars...")

        while not self._shutting_down:
            if self._shared_state.get("shutdown"):
                self._shutting_down = True
                break

            # ── Watchdog: check pending order timeouts ──────────────────
            # Skip in mock mode — time is compressed, wall-clock timeouts
            # don't apply. Mock fills arrive on the queue behind bars.
            timed_out = (
                [] if self._mock_client else self._orders.check_pending_timeouts()
            )
            if timed_out:
                for rec in timed_out:
                    msg_str = (
                        f"ORDER TIMEOUT {rec.order_id} ({rec.intent}) "
                        f"state={rec.state} reason={rec.reject_reason}"
                    )
                    logger.error(f"  {msg_str}")
                    self._gui.push(
                        {
                            "type": "ALERT",
                            "severity": "error",
                            "message": msg_str,
                        }
                    )
                # Force a position resync so we know what NT8 actually has
                from live.protocol import request_position

                try:
                    await self._client.send(request_position())
                except Exception:
                    pass

            # ── Surface any reconciliation mismatch from on_fill ────────
            if self._orders.last_reconcile_error:
                err = self._orders.last_reconcile_error
                self._orders.last_reconcile_error = ""
                self._gui.push(
                    {
                        "type": "ALERT",
                        "severity": "error",
                        "message": err,
                    }
                )

            # Dashboard requested a manual save
            if self._shared_state.get("save_now"):
                logger.info("  MANUAL SAVE requested from dashboard")
                self._periodic_save()
                self._shared_state["save_now"] = False

            # Dashboard manual order (FLATTEN/BUY/SELL)
            if self._shared_state.get("manual_order"):
                action = self._shared_state.pop("manual_order")
                logger.warning(f"  MANUAL {action} from dashboard")
                if action == "FLATTEN":
                    if not self._orders.is_flat:
                        order_msg = self._orders.build_exit_order(reason="manual")
                        if order_msg:
                            await self._client.send(order_msg)
                            self._orders.mark_sent("BAY_CLOSE")
                    if not self._pos_ledger.is_flat:
                        ledger_force_close(
                            self._pos_ledger,
                            self._last_price,
                            self._last_ts,
                            np.zeros(N_FEATURES),
                            reason="manual_flatten",
                        )

            # Stale bar detection (skip in mock mode — bars arrive instantly)
            # (monotonic clock, not wall time — independent of timezone/clock drift)
            last_arrival = getattr(self, "_last_arrival", 0)
            if last_arrival > 0 and not self._mock_client:
                silence_s = time.monotonic() - last_arrival
                if silence_s > 60 and self._broker_connected:
                    logger.error(
                        f"  STALE: {silence_s:.0f}s since last bar arrived — "
                        f"assuming NT8 panic, blocking new orders"
                    )
                    self._broker_connected = False
                elif silence_s < 15 and not self._broker_connected:
                    logger.warning(
                        f"  BARS FLOWING AGAIN — broker OK, unblocking orders"
                    )
                    self._broker_connected = True

            try:
                msg = await asyncio.wait_for(self._client.inbound.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # check shutdown flag every 1s

            msg_type = msg.get("type", "")
            if msg_type == MsgType.FILL:
                pnl_from_fill = self._orders.on_fill(msg)
                fill_px = float(msg.get("fill_price", 0))
                fill_time = float(msg.get("fill_time", time.time()))
                oid = msg.get("order_id", "")
                side = msg.get("side", "")

                # Look up the pending request context
                req = self._pending_requests.pop(oid, {})

                # Classify unknown fills (manual flatten, shutdown close, etc.)
                if not req:
                    if oid in ("Close", "BAY_CLOSE") or oid.startswith("Close"):
                        req = {
                            "type": "FLATTEN",
                            "tier": "MANUAL",
                            "direction": "",
                            "requested_price": fill_px,
                        }
                    else:
                        req = {
                            "type": "UNKNOWN",
                            "tier": "?",
                            "direction": "?",
                            "requested_price": fill_px,
                        }

                # ── Mutate ledger based on fill type ──────────────────
                req_type = req.get("type", "?")
                if req_type in ("ENTRY", "REENTRY"):
                    # OPEN fill → add position to ledger
                    ctx = req.get("entry_context", {})
                    feat = req.get("entry_features", np.zeros(N_FEATURES))
                    self._pos_ledger.add_position(
                        direction=req.get(
                            "direction", "long" if side == "BUY" else "short"
                        ),
                        entry_price=fill_px,
                        entry_ts=fill_time,
                        entry_tier=req.get("tier", "UNKNOWN"),
                        entry_features=feat,
                        is_chain=False,
                        cnn_flipped=req.get("cnn_flipped", False),
                        **ctx,
                    )
                    logger.info(
                        f"  LEDGER: added primary {req.get('direction')} "
                        f"{req.get('tier')} @ {fill_px:.2f}"
                    )

                elif req_type == "CHAIN_ENTRY":
                    # OPEN fill for chain → add chain to ledger
                    ctx = req.get("entry_context", {})
                    feat = req.get("entry_features", np.zeros(N_FEATURES))
                    self._pos_ledger.add_position(
                        direction=req.get(
                            "direction", "long" if side == "BUY" else "short"
                        ),
                        entry_price=fill_px,
                        entry_ts=fill_time,
                        entry_tier=req.get("tier", "UNKNOWN"),
                        entry_features=feat,
                        is_chain=True,
                        cnn_flipped=req.get("cnn_flipped", False),
                        **ctx,
                    )
                    logger.info(
                        f"  LEDGER: added chain {req.get('tier')} @ {fill_px:.2f}"
                    )

                elif req_type == "CHAIN_EXIT":
                    # REDUCE fill for chain → remove specific chain from ledger
                    cid = req.get("contract_id", "")
                    if cid and self._pos_ledger.get(cid):
                        self._pos_ledger.remove_position(
                            cid,
                            fill_px,
                            fill_time,
                            req.get("exit_reason", "chain_exit"),
                            np.zeros(N_FEATURES),
                        )
                        logger.info(f"  LEDGER: removed chain {cid}")
                    else:
                        logger.warning(
                            f"  LEDGER: chain {cid} not found for CHAIN_EXIT fill"
                        )

                elif req_type in ("EXIT", "NEGATIVE_EXIT"):
                    # REDUCE fill → close primary + all chains in ledger
                    cid = req.get("contract_id", "")
                    reason = req.get("exit_reason", "close")
                    # Close chains first, then primary
                    for chain in self._pos_ledger.chains:
                        self._pos_ledger.remove_position(
                            chain.contract_id,
                            fill_px,
                            fill_time,
                            f"chain_{reason}",
                            np.zeros(N_FEATURES),
                        )
                    if self._pos_ledger.primary is not None:
                        self._pos_ledger.remove_position(
                            self._pos_ledger.primary.contract_id,
                            fill_px,
                            fill_time,
                            reason,
                            np.zeros(N_FEATURES),
                        )
                    logger.info(f"  LEDGER: closed all positions ({reason})")

                elif req_type == "FLATTEN":
                    # Manual flatten or shutdown → close everything in ledger
                    ledger_force_close(
                        self._pos_ledger,
                        fill_px,
                        fill_time,
                        np.zeros(N_FEATURES),
                        reason="flatten_fill",
                    )
                    logger.info(f"  LEDGER: flattened all (manual/shutdown)")

                # ── Log FILL row with slippage ────────────────────────
                req_px = req.get("requested_price", fill_px)
                self._log_trade_event(
                    fill_time,
                    f"FILL_{req_type}",
                    req.get("tier", "?"),
                    req.get("direction", "?"),
                    req_px,
                    fill_px,
                    pnl_from_fill or 0,
                    0,
                    oid,
                    req.get("is_chain", False),
                )
                slip = fill_px - req_px
                if req.get("direction") == "short":
                    slip = -slip
                logger.info(
                    f"  FILL {oid} @ {fill_px:.2f} "
                    f"(requested {req_px:.2f}, slip={slip:+.2f})"
                )

                # Sync realized PnL from NT8
                self._daily_pnl = self._orders.daily_pnl
                continue
            elif msg_type == MsgType.ORDER_ACK:
                self._orders.on_order_ack(msg)
                continue
            elif msg_type == MsgType.TRADE_CLOSED:
                # NT8 ground-truth round-trip event
                self._on_nt8_trade_closed(msg)
                continue
            elif msg_type == "ORDER_STATUS":
                self._orders.on_order_status(msg)
                continue
            elif msg_type == "POSITION":
                self._orders.on_position(msg)
                # If NT8 says flat but ledger has positions, force-close
                # to keep state in sync. NT8 is ground truth.
                nt8_qty = int(msg.get("qty", 0))
                if nt8_qty == 0 and not self._pos_ledger.is_flat:
                    prim = self._pos_ledger.primary
                    logger.warning(
                        f"  POSITION sync: NT8 flat but ledger in "
                        f"{prim.direction if prim else '?'} "
                        f"{prim.entry_tier if prim else '?'} "
                        f"— forcing ledger flat"
                    )
                    ledger_force_close(
                        self._pos_ledger,
                        self._last_price,
                        self._last_ts,
                        np.zeros(N_FEATURES),
                        reason="nt8_position_sync",
                    )
                continue
            elif msg_type == MsgType.HEARTBEAT:
                # Enhanced heartbeat — reconcile position
                if "position_qty" in msg:
                    self._orders.on_heartbeat(msg)
                continue
            elif msg_type == "ACCOUNT_UPDATE":
                # NT8 is the source of truth for realized PnL
                self._nt8_realized_pnl = float(msg.get("realized_pnl", 0))
                self._nt8_unrealized_pnl = float(msg.get("unrealized_pnl", 0))
                self._nt8_cash_value = float(msg.get("cash_value", 0))
                # Push to dashboard
                self._gui.push(
                    {
                        "type": "ACCOUNT_UPDATE",
                        "cash_value": self._nt8_cash_value,
                        "realized_pnl": self._nt8_realized_pnl,
                        "unrealized_pnl": self._nt8_unrealized_pnl,
                    }
                )
                continue
            elif msg_type == "CONNECTION_LOST":
                self._broker_connected = False
                logger.error(
                    "  BROKER DISCONNECTED — blocking new orders, waiting for restore"
                )
                continue
            elif msg_type == "CONNECTION_RESTORED":
                self._broker_connected = True
                logger.warning("  BROKER RESTORED — requesting position snapshot")
                # Query actual NT8 position to reconcile after disconnect
                from live.protocol import request_position

                await self._client.send(request_position())
                continue
            elif msg_type == "MOCK_DONE":
                logger.info("  MockBridge: replay complete — shutting down")
                self._shutting_down = True
                break
            elif msg_type != MsgType.BAR:
                continue

            bar = self._extract_bar(msg)

            # Catch-up detection: measure how fast bars are arriving
            arrival_now = time.monotonic()
            inter_arrival = arrival_now - getattr(self, "_last_arrival", arrival_now)
            self._last_arrival = arrival_now
            if not hasattr(self, "_arrival_window"):
                self._arrival_window = []
            self._arrival_window.append(inter_arrival)
            if len(self._arrival_window) > 10:
                self._arrival_window.pop(0)
            avg_arrival = sum(self._arrival_window) / len(self._arrival_window)
            # Mock mode: never skip bars — process every one through the engine
            is_catchup = (
                not self._mock_client
                and len(self._arrival_window) >= 10
                and avg_arrival < 1.0
            )

            # Ingest: compute features + save (shared with Steps 4/5/6)
            feat = self._ingest_bar(bar)
            if feat is not None:
                z = feat[IDX_1M_Z_SE]
                rprob = feat[IDX_1M_RPROB]
                vel = feat[IDX_1M_VEL]
            else:
                continue

            if is_catchup:
                if self._bar_count % 100 == 0:
                    logger.info(
                        f"  CATCH-UP: {avg_arrival * 1000:.0f}ms inter-arrival "
                        f"(flooding {self._bar_count} bars)"
                    )
                continue  # skip engine + orders while backfilling

            # ── Evaluate via ledger + stateless engine ─────────────────
            # 1. Advance per-bar state on all open positions
            self._pos_ledger.update_bar(
                feat,
                bar["close"],
                bar["timestamp"],
                current_volume=bar.get("volume", 0),
            )

            # 2. Engine evaluates: pure function of (features, positions)
            was_flat = self._pos_ledger.is_flat
            eval_state = {
                "features_79d": feat,
                "price": bar["close"],
                "high": bar.get("high", bar["close"]),
                "low": bar.get("low", bar["close"]),
                "volume": bar.get("volume", 0),
                "timestamp": bar["timestamp"],
                "positions": self._pos_ledger.snapshot(),
            }
            # Hand the V2 getter to L5Decider (called lazily,
            # only when B7/B9 need V2 features).
            eval_state["v2_getter"] = getattr(self._lfe, "get_v2_vector", None)
            batch = self._engine.evaluate(eval_state)

            # 3. Apply counter updates (does NOT close/open positions)
            for pd in batch.position_decisions:
                self._pos_ledger.apply_position_decision(pd)

            events = []

            # 4. Process per-position exits → send NT8 orders
            #    (ledger is NOT mutated here — only on FILL confirmation)
            for exit_sig in batch.exits:
                pos = self._pos_ledger.get(exit_sig.contract_id)
                if pos is None:
                    continue
                if pos.is_chain:
                    # Chain exit → scale-out
                    if self._orders.can_scale_out and self._broker_connected:
                        order_msg = self._orders.build_scale_out_order(
                            reason=exit_sig.reason
                        )
                        if order_msg:
                            self._pending_requests[order_msg["order_id"]] = {
                                "requested_price": bar["close"],
                                "tier": pos.entry_tier,
                                "direction": pos.direction,
                                "is_chain": True,
                                "type": "CHAIN_EXIT",
                                "contract_id": exit_sig.contract_id,
                                "exit_reason": exit_sig.reason,
                            }
                            await self._client.send(order_msg)
                            self._orders.mark_sent(order_msg["order_id"])
                            events.append(f"CHAIN_EXIT_{exit_sig.reason}")
                            self._log_trade_event(
                                bar["timestamp"],
                                "CHAIN_EXIT",
                                pos.entry_tier,
                                pos.direction,
                                bar["close"],
                                0,
                                0,
                                pos.bars_held,
                                exit_sig.reason,
                                True,
                            )
                            logger.info(f"  CHAIN EXIT {exit_sig.reason}")
                else:
                    # Primary exit → close all remaining (only if we can
                    # actually send the order — otherwise it's a duplicate
                    # signal for an already-pending exit)
                    if self._orders.can_exit and self._broker_connected:
                        order_msg = self._orders.build_exit_order(
                            reason=exit_sig.reason
                        )
                        if order_msg:
                            order_msg["requested_price"] = float(bar["close"])
                            self._pending_requests["BAY_CLOSE"] = {
                                "requested_price": bar["close"],
                                "tier": pos.entry_tier,
                                "direction": pos.direction,
                                "is_chain": False,
                                "type": "EXIT",
                                "contract_id": exit_sig.contract_id,
                                "exit_reason": exit_sig.reason,
                            }
                            await self._client.send(order_msg)
                            self._orders.mark_sent("BAY_CLOSE")
                            events.append(f"EXIT_{exit_sig.reason}")
                            self._log_trade_event(
                                bar["timestamp"],
                                "EXIT",
                                pos.entry_tier,
                                pos.direction,
                                bar["close"],
                                0,
                                0,
                                pos.bars_held,
                                exit_sig.reason,
                                False,
                            )
                            logger.info(f"  EXIT {exit_sig.reason}")

            # 5. Negative exit → close primary + all chains
            if batch.negative_exit is not None:
                reason = batch.negative_exit.reason
                prim = self._pos_ledger.primary
                if prim and self._orders.can_exit and self._broker_connected:
                    order_msg = self._orders.build_exit_order(reason=reason)
                    if order_msg:
                        self._pending_requests["BAY_CLOSE"] = {
                            "requested_price": bar["close"],
                            "tier": prim.entry_tier,
                            "direction": prim.direction,
                            "is_chain": False,
                            "type": "NEGATIVE_EXIT",
                            "contract_id": prim.contract_id,
                            "exit_reason": reason,
                        }
                        await self._client.send(order_msg)
                        self._orders.mark_sent("BAY_CLOSE")
                        events.append(f"NEG_EXIT_{reason}")
                        self._log_trade_event(
                            bar["timestamp"],
                            "NEGATIVE_EXIT",
                            prim.entry_tier,
                            prim.direction,
                            bar["close"],
                            0,
                            0,
                            prim.bars_held,
                            reason,
                            False,
                        )
                        logger.info(f"  NEGATIVE EXIT {reason}")

            # 6. Chain entry → scale-in order
            if (
                batch.chain_entry is not None
                and not self._pos_ledger.is_flat
                and self._orders.can_scale_in
                and self._broker_connected
            ):
                sig = batch.chain_entry
                side = "BUY" if sig.direction == "long" else "SELL"
                order_msg = self._orders.build_scale_in_order(side)
                if order_msg:
                    ctx = _compute_entry_context(feat, sig.direction)
                    self._pending_requests[order_msg["order_id"]] = {
                        "requested_price": bar["close"],
                        "tier": sig.tier,
                        "direction": sig.direction,
                        "is_chain": True,
                        "type": "CHAIN_ENTRY",
                        "cnn_flipped": sig.cnn_flipped,
                        "entry_features": feat.copy(),
                        "entry_context": ctx,
                    }
                    await self._client.send(order_msg)
                    self._orders.mark_sent(order_msg["order_id"])
                    events.append(f"CHAIN_ENTRY_{sig.tier}")
                    self._log_trade_event(
                        bar["timestamp"],
                        "CHAIN_ENTRY",
                        sig.tier,
                        sig.direction,
                        bar["close"],
                        0,
                        0,
                        0,
                        "",
                        True,
                    )
                    logger.info(f"  CHAIN ENTRY {sig.tier} @ {bar['close']:.2f}")

            # 7. Fresh entry. Normally requires ledger flat -- but for an
            # L5 zigzag-reverse flip, the engine signals exit + new-direction
            # entry in the SAME batch. In that case the exit order has just
            # been sent (FILL not yet processed); we send the entry order
            # immediately after so they execute back-to-back at the same
            # bar's close. NT8 SIM serializes this; mock fills both at
            # requested_price.
            entry_sig = batch.entry
            is_flip = entry_sig is not None and any(
                d.exit_reason for d in batch.position_decisions
            )
            # On flip, bypass `can_enter` (which requires is_flat) since the
            # exit order was just sent and will land before the entry fill.
            if (
                entry_sig is not None
                and (self._pos_ledger.is_flat or is_flip)
                and (self._orders.can_enter or is_flip)
                and self._broker_connected
            ):
                side = "BUY" if entry_sig.direction == "long" else "SELL"
                order_msg = self._orders.build_entry_order(side, allow_flip=is_flip)
                if order_msg:
                    # Stamp the bar close where the decision was made so the
                    # mock bridge can fill exactly there (zero-slip integration test).
                    order_msg["requested_price"] = float(bar["close"])
                    ctx = _compute_entry_context(feat, entry_sig.direction)
                    self._pending_requests[order_msg["order_id"]] = {
                        "requested_price": bar["close"],
                        "tier": entry_sig.tier,
                        "direction": entry_sig.direction,
                        "is_chain": False,
                        "type": "ENTRY",
                        "cnn_flipped": entry_sig.cnn_flipped,
                        "entry_features": feat.copy(),
                        "entry_context": ctx,
                    }
                    await self._client.send(order_msg)
                    self._orders.mark_sent(order_msg["order_id"])
                events.append(f"ENTRY_{entry_sig.tier}")
                self._log_trade_event(
                    bar["timestamp"],
                    "ENTRY",
                    entry_sig.tier,
                    entry_sig.direction,
                    bar["close"],
                    0,
                    0,
                    0,
                    "",
                    False,
                )
                logger.info(
                    f"  ENTRY {entry_sig.direction} "
                    f"{entry_sig.tier} @ {bar['close']:.2f}"
                )

            # 8. Fast re-evaluation: if exits made us flat, re-evaluate
            #    for entry on the same bar (mirrors sim_executor behavior).
            if not was_flat and self._pos_ledger.is_flat:
                eval_state["positions"] = self._pos_ledger.snapshot()
                batch2 = self._engine.evaluate(eval_state)
                if (
                    batch2.entry is not None
                    and self._pos_ledger.is_flat
                    and self._orders.can_enter
                    and self._broker_connected
                ):
                    sig2 = batch2.entry
                    side2 = "BUY" if sig2.direction == "long" else "SELL"
                    order_msg = self._orders.build_entry_order(side2)
                    if order_msg:
                        ctx2 = _compute_entry_context(feat, sig2.direction)
                        self._pending_requests[order_msg["order_id"]] = {
                            "requested_price": bar["close"],
                            "tier": sig2.tier,
                            "direction": sig2.direction,
                            "is_chain": False,
                            "type": "REENTRY",
                            "cnn_flipped": sig2.cnn_flipped,
                            "entry_features": feat.copy(),
                            "entry_context": ctx2,
                        }
                        await self._client.send(order_msg)
                        self._orders.mark_sent(order_msg["order_id"])
                    events.append(f"REENTRY_{sig2.tier}")
                    logger.info(
                        f"  REENTRY {sig2.direction} {sig2.tier} @ {bar['close']:.2f}"
                    )

            # ── PnL tracking ───────────────────────────────────────────
            self._daily_pnl = getattr(self, "_nt8_realized_pnl", None)
            if self._daily_pnl is None:
                self._daily_pnl = self._orders.daily_pnl
            self._trade_count = len(self._pos_ledger.closed_trades)

            # Unrealized: NT8 ACCOUNT_UPDATE is truth, fallback to ledger
            unrealized = self._nt8_unrealized_pnl
            if unrealized == 0 and not self._pos_ledger.is_flat:
                px = bar["close"]
                for pos in [self._pos_ledger.primary] + self._pos_ledger.chains:
                    if pos is None:
                        continue
                    if pos.direction == "long":
                        unrealized += (px - pos.entry_price) / TICK * TV
                    else:
                        unrealized += (pos.entry_price - px) / TICK * TV

            n_contracts = self._pos_ledger.n_contracts
            event_str = " ".join(events)

            # ── Write ledger row ───────────────────────────────────────
            self._write_ledger(
                bar["timestamp"],
                bar["close"],
                z,
                rprob,
                vel,
                unrealized,
                event_str,
                n_contracts,
            )

            # ── Dashboard ──────────────────────────────────────────────
            prim = self._pos_ledger.primary
            in_pos = prim is not None
            direction = prim.direction if prim else ""
            tier = prim.entry_tier if prim else ""
            self._gui.push(
                {
                    "type": "TICK_UPDATE",
                    "price": bar["close"],
                    "bars": self._bar_count,
                    "unrealized": unrealized,
                    "daily_pnl": self._daily_pnl,
                    "in_position": in_pos,
                    "direction": direction,
                    "tier": tier,
                    "entry_price": prim.entry_price if prim else 0.0,
                    "z_se": z,
                    "rprob": rprob,
                    "is_1m": False,
                }
            )

            # Push the decider's internal state so the dashboard
            # can overlay the indicators that actually drive decisions
            # (zigzag extreme, R-trigger threshold, B7/B9/B10).
            eng = self._engine
            zz = getattr(eng, "_zz_direction", 0)
            self._gui.push(
                {
                    "type": "L5_STATE",
                    "r_price": getattr(eng, "_r_price", None),
                    "zz_dir": ("up" if zz == 1 else "down" if zz == -1 else None),
                    "zz_extreme": getattr(eng, "_zz_extreme_price", None),
                    "b10_mode": getattr(eng, "_b10_mode", None),
                    "b7_pred": getattr(eng, "_last_b7_pred", None),
                    "b9_pred": getattr(eng, "_last_b9_pred", None),
                }
            )
            # A detected-but-skipped R-trigger: surface it on the chart.
            skip = getattr(eng, "_this_bar_skip", None)
            if skip is not None:
                self._gui.push(
                    {
                        "type": "SIGNAL_SKIP",
                        "price": bar["close"],
                        "direction": skip.get("direction", "?"),
                        "reason": skip.get("reason", ""),
                        "pred_R": skip.get("pred_R"),
                        "thr": skip.get("thr"),
                        "skip_count": skip.get("skip_count", 0),
                    }
                )
            # Mirror the L5 state to a flat file the NT8 BayesianCompanion
            # indicator reads -- so the NT8 chart shows what Python shows.
            self._write_l5_overlay(bar, skip, events)

            for ev in events:
                if "ENTRY" in ev:
                    is_chain = "CHAIN" in ev
                    label = "CHAIN_ENTRY" if is_chain else "ENTRY"
                    self._gui.push_trade_marker(
                        label, "BUY" if direction == "long" else "SELL", bar["close"]
                    )
                elif "EXIT" in ev:
                    is_chain = "CHAIN" in ev
                    closed = self._pos_ledger.closed_trades
                    # Find the just-closed trade (last one matching is_chain)
                    matching = [
                        t for t in closed if t.get("is_chain", False) == is_chain
                    ]
                    last_t = matching[-1] if matching else None
                    last_pnl = last_t["pnl"] if last_t else 0
                    entry_px = last_t.get("entry_price", 0) if last_t else 0
                    label = "CHAIN_EXIT" if is_chain else "EXIT"
                    self._gui.push_trade_marker(label, "", bar["close"], pnl=last_pnl)
                    # Push as engine trade so dashboard log shows it
                    # (NT8_TRADE only fires for live NT8 sessions; mock/sim
                    # need this for the trade log to populate.)
                    self._gui.push(
                        {
                            "type": "NT8_TRADE",
                            "side": (last_t.get("dir", "") if last_t else "").upper()
                            or "",
                            "entry_price": entry_px,
                            "exit_price": bar["close"],
                            "pnl": last_pnl,
                            "fill_time": bar["timestamp"],
                            "is_chain": is_chain,
                            "order_id": f"ENG_{ev}",
                        }
                    )

            closed = self._pos_ledger.closed_trades
            wins = sum(1 for t in closed if t["pnl"] > 0)
            gross_win = sum(t["pnl"] for t in closed if t["pnl"] > 0)
            gross_loss = sum(t["pnl"] for t in closed if t["pnl"] <= 0)
            z_pct = min(abs(z) / 2.0 * 100, 100)
            eb = {
                "reversed": 0,
                "q1": 0,
                "q2": 0,
                "q3": 0,
                "q4": 0,
                "q100plus": 0,
                "forced": 0,
            }
            self._gui.push_stats(
                session_pnl=self._daily_pnl,
                session_wins=wins,
                session_trades=self._trade_count,
                gross_win=gross_win,
                gross_loss=gross_loss,
                exit_buckets=eb,
                belief_pct=z_pct,
                in_position=in_pos,
                daily_pnl=self._daily_pnl,
            )

            # ── Status line ────────────────────────────────────────────
            if self._feat_count % 12 == 0:
                n_chains = len(self._pos_ledger.chains)
                pos = direction or "FLAT"
                chain_str = f"+{n_chains}ch" if n_chains else ""
                print(
                    f"\r  {self._ts_str(bar['timestamp'])} | {bar['close']:>10.2f} | "
                    f"{pos:>5} {tier:>15} {chain_str:>4} | z={z:>+5.1f} rprob={rprob:.2f} | "
                    f"tr={self._trade_count} day=${self._daily_pnl:>+.0f}    ",
                    end="",
                    flush=True,
                )

            # ── Position sync ──────────────────────────────────────────
            # Every 15s (3 bars @ 5s): request NT8 position for reconciliation
            # (periodic save already happens in _ingest_bar)
            if self._bar_count % 3 == 0:
                from live.protocol import request_position

                await self._client.send(request_position())

            # ── Engine state push (every bar) ──────────────────────────
            self._push_engine_state()

            # Stale order cleanup every 5 min
            if self._bar_count % 60 == 0:
                self._orders.cleanup_stale_orders(max_age_s=120.0)

    # ═══════════════════════════════════════════════════════════════════
    # TRADE LOG — one row per entry/exit for parity comparison
    # ═══════════════════════════════════════════════════════════════════

    def _on_nt8_trade_closed(self, msg):
        """Handle TRADE_CLOSED — NT8 ground-truth round trip."""
        order_id = msg.get("order_id", "")
        side = msg.get("side", "")
        entry = float(msg.get("entry_price", 0))
        exit_p = float(msg.get("exit_price", 0))
        pnl = float(msg.get("pnl", 0))
        fill_time = float(msg.get("fill_time", time.time()))
        qty = int(msg.get("qty", 1))
        is_chain = bool(msg.get("is_chain", False))

        # Write to NT8 ground-truth log
        if self._nt8_trade_log:
            row = [
                f"{fill_time:.0f}",
                order_id,
                side,
                f"{entry:.2f}",
                f"{exit_p:.2f}",
                f"{pnl:.2f}",
                str(qty),
                "1" if is_chain else "0",
            ]
            self._nt8_trade_log.write(",".join(row) + "\n")
            self._nt8_trade_log.flush()

        # Push to dashboard trade log + trade marker
        self._gui.push(
            {
                "type": "NT8_TRADE",
                "order_id": order_id,
                "side": side,
                "entry_price": entry,
                "exit_price": exit_p,
                "pnl": pnl,
                "fill_time": fill_time,
                "is_chain": is_chain,
            }
        )
        self._gui.push_trade_marker("NT8_EXIT", side, exit_p, pnl=pnl)

        logger.info(
            f"  NT8 TRADE: {order_id} {side} "
            f"{entry:.2f}→{exit_p:.2f} pnl=${pnl:+.2f}"
            f"{' [chain]' if is_chain else ''}"
        )

    def _push_engine_state(self):
        """Push engine health to dashboard — state, bar flow, activity."""
        # Derive state
        if self._shutting_down:
            state = "SHUTDOWN"
        elif not self._broker_connected:
            state = "BROKER_DISCONNECTED"
        elif not self._synced:
            state = "SYNCING"
        elif self._last_ts == 0:
            state = "WARMUP"
        else:
            # Check if bars are fresh (inter-arrival < 10s avg)
            window = getattr(self, "_arrival_window", [])
            if len(window) >= 5:
                avg_arrival = sum(window) / len(window)
                if avg_arrival < 1.0:
                    state = "CATCH_UP"
                elif avg_arrival > 30:
                    state = "STALE"
                else:
                    state = "TRADING"
            else:
                state = "TRADING"

        # Bars per minute from arrival window
        bar_rate = 0.0
        window = getattr(self, "_arrival_window", [])
        if len(window) >= 3:
            avg_s = sum(window) / len(window)
            if avg_s > 0:
                bar_rate = 60.0 / avg_s

        # Activity description
        activity = ""
        prim = self._pos_ledger.primary if self._pos_ledger else None
        if prim is not None:
            n_chains = len(self._pos_ledger.chains)
            chain_str = f" +{n_chains}ch" if n_chains else ""
            activity = f"{prim.direction} {prim.entry_tier}{chain_str}"

        self._gui.push(
            {
                "type": "ENGINE_STATE",
                "state": state,
                "bar_count": self._bar_count,
                "last_bar_ts": self._last_ts,
                "bar_rate": bar_rate,
                "activity": activity,
            }
        )

    def _log_trade_event(
        self,
        ts,
        event_type,
        tier,
        direction,
        requested_price,
        fill_price,
        pnl,
        bars_held,
        exit_reason,
        is_chain,
    ):
        """Write one trade event row with slippage tracking.

        requested_price: what the engine saw when it decided to trade
        fill_price: what NT8 actually filled at (0 if not yet filled)
        """
        if self._trade_log is None:
            return
        slip = fill_price - requested_price if fill_price > 0 else 0
        # Normalize: positive slippage = worse fill
        if direction == "short" or (direction == "" and event_type.startswith("EXIT")):
            slip = -slip  # for shorts, higher fill on entry = worse
        row = [
            f"{ts:.0f}",
            event_type,
            tier,
            direction,
            f"{requested_price:.2f}",
            f"{fill_price:.2f}",
            f"{slip:.2f}",
            f"{pnl:.1f}",
            str(bars_held),
            exit_reason,
            "1" if is_chain else "0",
            str(self._pos_ledger.n_contracts),
            f"{self._daily_pnl:.1f}",
        ]
        self._trade_log.write(",".join(row) + "\n")
        self._trade_log.flush()

    # ═══════════════════════════════════════════════════════════════════
    # LEDGER + SAVE
    # ═══════════════════════════════════════════════════════════════════

    def _write_ledger(self, ts, price, z, rprob, vel, pnl, event, n_contracts=0):
        if self._ledger is None:
            return
        prim = self._pos_ledger.primary
        row = [
            f"{ts:.0f}",
            f"{price:.2f}",
            f"{z:.4f}",
            f"{rprob:.4f}",
            f"{vel:.2f}",
            "1" if prim is not None else "0",
            prim.direction if prim else "",
            prim.entry_tier if prim else "",
            str(prim.bars_held if prim else 0),
            f"{pnl:.1f}",
            f"{prim.peak_pnl if prim else 0:.1f}",
            str(n_contracts),
            event,
        ]
        self._ledger.write(",".join(row) + "\n")

    def _periodic_save(self):
        if self._ledger and not self._ledger.closed:
            self._ledger.flush()
        # Save checkpoint (velocities + ledger state for recovery)
        if self._lfe:
            # Serialize ledger positions for crash recovery
            ledger_state = {"positions": []}
            if self._pos_ledger and not self._pos_ledger.is_flat:
                for pos in [self._pos_ledger.primary] + self._pos_ledger.chains:
                    if pos is None:
                        continue
                    ledger_state["positions"].append(
                        {
                            "contract_id": pos.contract_id,
                            "direction": pos.direction,
                            "entry_price": pos.entry_price,
                            "entry_ts": pos.entry_ts,
                            "entry_tier": pos.entry_tier,
                            "is_chain": pos.is_chain,
                            "cnn_flipped": pos.cnn_flipped,
                            "bars_held": pos.bars_held,
                            "peak_pnl": pos.peak_pnl,
                            "entry_abs_z": pos.entry_abs_z,
                            "entry_velocity": pos.entry_velocity,
                            "entry_h1_z": pos.entry_h1_z,
                            "entry_vol_rel": pos.entry_vol_rel,
                            "ride_exit_bars": pos.ride_exit_bars,
                        }
                    )
            os.makedirs(os.path.dirname(LIVE_CHECKPOINT) or ".", exist_ok=True)
            import json as _json

            cp = {
                "version": 5,
                "last_ts": self._last_ts,
                "velocities": self._lfe.prev_velocities,
                "accumulators": self._lfe._accumulators,
                "ledger_state": ledger_state,
                "bar_counts": self._lfe.bar_counts,
                # Parity metadata: what ATLAS_NT8 data the LFE loaded at warmup.
                # The parity tool uses this to truncate warmup to the same point.
                "preloaded_end_ts": self._lfe._last_loaded_ts,
                "day_ends": self._lfe._day_ends,
            }
            with open(LIVE_CHECKPOINT, "w", encoding="utf-8") as f:
                _json.dump(cp, f)
        # Save live features — same schema as build_dataset output
        self._save_live_features()
        # Save live bars to ATLAS_LIVE
        if self._live_bars:
            from live.incremental_writer import IncrementalWriter

            writer = IncrementalWriter(ATLAS_LIVE, self._session_date)
            writer.save_all_chunks({"5s": self._live_bars})

    def _save_live_features(self):
        """Save accumulated 91D features to parquet — same format as build_dataset."""
        if not self._live_79d:
            return
        from core_v2.features import FEATURE_NAMES

        os.makedirs(FEATURES_LIVE, exist_ok=True)

        # Group by UTC date
        by_date = {}
        for row in self._live_79d:
            ts = row["timestamp"]
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y_%m_%d")
            by_date.setdefault(day, []).append(row)

        for day, rows in by_date.items():
            timestamps = [r["timestamp"] for r in rows]
            features = np.array([r["features"] for r in rows])
            data = {"timestamp": timestamps}
            for i, name in enumerate(FEATURE_NAMES):
                data[name] = features[:, i] if i < features.shape[1] else 0.0
            df = pd.DataFrame(data)
            df["timestamp"] = df["timestamp"].astype(np.int64)
            out_path = os.path.join(FEATURES_LIVE, f"{day}.parquet")
            df.to_parquet(out_path, index=False)
        logger.info(f"  Live features: {len(self._live_79d)} rows -> {FEATURES_LIVE}/")

    # ═══════════════════════════════════════════════════════════════════
    # SHUTDOWN
    # ═══════════════════════════════════════════════════════════════════

    async def _shutdown(self):
        self._shutting_down = True
        logger.info("")
        logger.info("SHUTDOWN")

        # 1. Close positions (need _orders and _pos_ledger still alive)
        if self._orders and not self._orders.is_flat:
            order_msg = self._orders.build_exit_order(reason="shutdown")
            if order_msg and self._client:
                await self._client.send(order_msg)
            await asyncio.sleep(2.0)
        if self._pos_ledger and not self._pos_ledger.is_flat:
            ledger_force_close(
                self._pos_ledger,
                self._last_price,
                self._last_ts,
                np.zeros(N_FEATURES),
                reason="shutdown",
            )

        # 2. Final save (need _lfe and _engine still alive)
        try:
            self._periodic_save()
            logger.info(f"  Checkpoint saved: {LIVE_CHECKPOINT}")
        except Exception as e:
            logger.error(f"  Checkpoint save failed: {e}")

        # 3. Close file handles
        if self._ledger:
            try:
                self._ledger.flush()
                self._ledger.close()
                logger.info(f"  Ledger: {self._ledger_path}")
            except Exception:
                pass
        if hasattr(self, "_trade_log") and self._trade_log:
            try:
                self._trade_log.flush()
                self._trade_log.close()
                logger.info(f"  Trades: {self._trade_log_path}")
            except Exception:
                pass
        if hasattr(self, "_nt8_trade_log") and self._nt8_trade_log:
            try:
                self._nt8_trade_log.flush()
                self._nt8_trade_log.close()
                logger.info(f"  NT8 trades: {self._nt8_trade_log_path}")
            except Exception:
                pass

        # 4. Summary (uses ledger)
        wins = 0
        chains = 0
        if self._pos_ledger:
            closed = self._pos_ledger.closed_trades
            wins = sum(1 for t in closed if t["pnl"] > 0)
            chains = sum(
                1 for t in closed if str(t.get("exit_reason", "")).startswith("chain_")
            )
        logger.info(f"  Bars:     {self._bar_count:,}")
        logger.info(f"  Feats:    {self._feat_count:,}")
        logger.info(f"  Trades:   {self._trade_count} ({chains} chains)")
        logger.info(
            f"  Win rate: {wins}/{self._trade_count} "
            f"({wins / max(self._trade_count, 1) * 100:.0f}%)"
        )
        logger.info(f"  PnL:      ${self._daily_pnl:.0f}")

        # 5. Disconnect (with timeout — socket may be dead)
        if self._client:
            try:
                await asyncio.wait_for(self._client.disconnect(), timeout=5.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"  Disconnect failed/timed out: {e}")

        # 6. Release GPU + RAM (AFTER everything else uses engine/lfe)
        # cuda.close() is notorious for hanging when contexts have dangling
        # references. Run it in a background thread with a hard timeout so
        # shutdown is never blocked by the GPU release.
        def _close_cuda():
            try:
                from numba import cuda

                if cuda.is_available():
                    cuda.close()
            except Exception as e:
                logger.warning(f"  cuda.close() failed: {e}")

        import threading

        t = threading.Thread(target=_close_cuda, daemon=True)
        t.start()
        t.join(timeout=CUDA_CLOSE_TIMEOUT_S)
        if t.is_alive():
            logger.warning(
                f"  cuda.close() timed out after {CUDA_CLOSE_TIMEOUT_S}s "
                f"— continuing shutdown (CUDA context will leak until process exit)"
            )

        self._lfe = None
        self._engine = None
        self._live_bars.clear()
        self._live_79d.clear()
        import gc

        gc.collect()
        logger.info("  Memory released")
        logger.info("  Done")

        # Final hammer: force-exit the process if anything (including the
        # tk dashboard thread or a non-daemon NT8 client thread) is still
        # holding the interpreter alive after a clean shutdown. We've
        # already saved state, closed positions, and flushed logs.
        os._exit(0)

    # ═══════════════════════════════════════════════════════════════════
    # BAR PROCESSING — shared across all steps
    # ═══════════════════════════════════════════════════════════════════

    def _ingest_bar(self, bar: dict) -> "Optional[np.ndarray]":
        """Process one 5s bar: feed to LFE, save features + bar data.

        Called from Steps 4/5/6/7. Always accumulates bars and features
        so the live dataset builds continuously from first bar received.
        Returns the feature vector if computed, None if duplicate/skip.
        """
        ts = bar["timestamp"]
        if ts > self._last_ts:
            self._last_ts = ts
            self._last_price = bar["close"]

        # Feed to LFE (dedupes internally)
        feat = self._lfe.on_bar(bar)
        if feat is None:
            return None

        # Dedup feature save
        last_saved_ts = self._live_79d[-1]["timestamp"] if self._live_79d else 0
        if ts <= last_saved_ts:
            return feat

        # Accumulate bar + features for periodic save
        self._bar_count += 1
        self._feat_count += 1
        self._live_bars.append(bar)
        self._live_79d.append({"timestamp": ts, "features": feat.copy()})

        # Periodic save every 3 bars (15s) in live, every 500 bars in mock
        save_interval = 500 if self._mock_client else 3
        if self._bar_count % save_interval == 0:
            self._periodic_save()

        return feat

    # L5 overlay file for the NT8 BayesianCompanion indicator
    _L5_OVERLAY_PATH = "live/state/l5_overlay.txt"

    def _write_l5_overlay(self, bar, skip, events):
        """Write the current L5 state to a flat file the NT8 companion
        indicator reads. Atomic write (tmp + replace). One line per field;
        SKIP/TRADE event lines are pipe-delimited. Best-effort -- never
        let an overlay-write failure disrupt trading.
        """
        try:
            eng = self._engine
            zz = getattr(eng, "_zz_direction", 0)
            zz_dir = "up" if zz == 1 else "down" if zz == -1 else ""
            r_price = getattr(eng, "_r_price", None) or 0.0
            ext = getattr(eng, "_zz_extreme_price", None) or 0.0
            thr = 0.0
            if r_price and ext and zz_dir:
                thr = (ext - r_price) if zz_dir == "up" else (ext + r_price)
            prim = self._pos_ledger.primary
            b7 = getattr(eng, "_last_b7_pred", None)
            b9 = getattr(eng, "_last_b9_pred", None)
            ext_ts = getattr(eng, "_zz_extreme_ts", None) or 0
            # Confirmed zigzag pivots -> the line the NT8 companion draws.
            # Compact: ts:price|ts:price|...
            pivots = getattr(eng, "_zz_pivots", None)
            if pivots:
                zz_pivots = "|".join(
                    f"{int(pt)}:{px:.2f}" for (pt, px, _kind) in pivots
                )
            else:
                zz_pivots = ""
            lines = [
                f"ts={int(bar['timestamp'])}",
                f"price={bar['close']}",
                f"r_price={r_price:.2f}",
                f"zz_dir={zz_dir}",
                f"zz_extreme={ext:.2f}",
                f"zz_extreme_ts={int(ext_ts)}",
                f"zz_threshold={thr:.2f}",
                f"zz_pivots={zz_pivots}",
                f"b10_mode={getattr(eng, '_b10_mode', '') or ''}",
                f"b7={b7:.3f}" if b7 is not None else "b7=",
                f"b9={b9:.1f}" if b9 is not None else "b9=",
                f"skip_count={getattr(eng, '_skip_count', 0)}",
                f"position={'flat' if prim is None else prim.direction}",
                f"pos_entry={prim.entry_price if prim else 0.0}",
                f"day_pnl={self._daily_pnl:.0f}",
            ]
            if skip is not None:
                lines.append(
                    f"SKIP|{int(bar['timestamp'])}|{bar['close']}|"
                    f"{skip.get('direction', '?')}"
                )
            for ev in events:
                if "ENTRY" in ev:
                    # Entry direction = this bar's R-trigger direction
                    # (the ledger primary isn't filled yet at this point).
                    side = getattr(eng, "_this_bar_rtrig_dir", "") or (
                        prim.direction if prim else ""
                    )
                    lines.append(
                        f"TRADE|{int(bar['timestamp'])}|{bar['close']}|{side}|ENTRY"
                    )
                elif "EXIT" in ev:
                    lines.append(f"TRADE|{int(bar['timestamp'])}|{bar['close']}||EXIT")
            os.makedirs(os.path.dirname(self._L5_OVERLAY_PATH), exist_ok=True)
            tmp = self._L5_OVERLAY_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            os.replace(tmp, self._L5_OVERLAY_PATH)
        except Exception as e:
            logger.debug(f"L5 overlay write skipped: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_bar(msg):
        return {
            "timestamp": msg.get("timestamp", 0),
            "open": msg.get("open", 0),
            "high": msg.get("high", 0),
            "low": msg.get("low", 0),
            "close": msg.get("close", 0),
            "volume": msg.get("volume", 0),
        }

    @staticmethod
    def _ts_str(ts):
        if ts < 1e9:
            return "?"
        # Local time so startup logs match the Python logging prefix (which
        # is also local). Matches wall-clock the user sees on their machine.
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# NT8 accounts the engine runs against without an extra confirmation --
# all simulation / replay accounts. Playback101 is the account NT8 Market
# Replay (playback) trades into.
KNOWN_SIM_ACCOUNTS = ("Sim101", "Sim102", "Playback101", "DEMO6872628")


def _resolve_account(args) -> str:
    """Decide which NT8 account this run trades into.

    Enforces a strict whitelist for simulation/replay accounts. If a real
    money account is requested via --account, it requires interactive
    confirmation. If headless, real money accounts are hard-refused.
    """
    default = LiveConfig().account
    acct = args.account

    if not acct:
        if args.mock:
            return default  # mock bridge: account is cosmetic
        if args.headless:
            raise SystemExit(
                "engine_v2: --account is required with --headless "
                "(e.g. --account Playback101 for NT8 Market Replay)."
            )

        print()
        print("  NT8 account for this run")
        print(f"    known sim/replay accounts: {', '.join(KNOWN_SIM_ACCOUNTS)}")
        print("    Playback101 = NT8 Market Replay (playback)")
        acct = input(f"  account [{default}]: ").strip() or default

    if acct not in KNOWN_SIM_ACCOUNTS:
        if args.headless:
            raise SystemExit(
                f'engine_v2: HARD REFUSE. "{acct}" is not a '
                f"known sim account {KNOWN_SIM_ACCOUNTS}. Cannot "
                f"trade a real account in --headless mode."
            )

        confirm = (
            input(
                f'  !!! "{acct}" is NOT a known sim/replay account !!!\n'
                f"  Are you SURE you want to trade real money? [y/N]: "
            )
            .strip()
            .lower()
        )
        if confirm != "y":
            raise SystemExit("engine_v2: aborted — relaunch with a sim/replay account.")

    return acct


def main():
    parser = argparse.ArgumentParser(description="Live Engine V2")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--headless", action="store_true", help="No dashboard GUI")
    parser.add_argument(
        "--mock", action="store_true", help="Use MockBridge (replay ATLAS bars)"
    )
    parser.add_argument(
        "--mock-day", type=str, default=None, help="Day to replay (e.g. 2026_04_16)"
    )
    parser.add_argument(
        "--pivot-source",
        choices=["stream", "replay"],
        default="stream",
        help="L5: stream (forward pass detector) or replay "
        "(inject pivots from production parquet, for SIM validation)",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="NT8 account (e.g. Sim101, Playback101). Prompted at startup if omitted.",
    )
    args = parser.parse_args()

    account = _resolve_account(args)

    # LiveConfig is frozen; rebuild with overrides
    import dataclasses

    config = dataclasses.replace(
        LiveConfig(), pivot_source=args.pivot_source, account=account
    )
    logger.info(
        f"Engine: L5Decider  pivot_source={config.pivot_source}  "
        f"account={config.account}"
    )
    shared_state = {}
    gui_queue = None

    # Launch dashboard unless headless
    if not args.headless:
        import queue as stdlib_queue
        import threading
        import tkinter as tk

        gui_queue = stdlib_queue.Queue(maxsize=5000)

        def _run_dashboard():
            try:
                from visualization.dashboard_v2 import TradingDashboard

                root = tk.Tk()
                popup = TradingDashboard(root, gui_queue, shared_state=shared_state)

                def _on_close():
                    shared_state["shutdown"] = True
                    root.destroy()

                root.protocol("WM_DELETE_WINDOW", _on_close)
                root.mainloop()
            except Exception as e:
                logger.warning(f"Dashboard failed: {e}")

        t = threading.Thread(target=_run_dashboard, daemon=True)
        t.start()
        logger.info("Dashboard launched")

    mock_client = None
    if args.mock:
        from live.mock_bridge import MockBridge

        mock_client = MockBridge(config, day=args.mock_day)
        logger.info(f"Mock mode: replaying {args.mock_day or 'latest'} from ATLAS_NT8")

    engine = LiveEngineV2(
        config,
        skip_check=args.skip_check,
        skip_build=args.skip_build,
        gui_queue=gui_queue,
        shared_state=shared_state,
        mock_client=mock_client,
    )
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
