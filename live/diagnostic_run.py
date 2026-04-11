"""
Diagnostic Run — single connection, 4 sequential phases.

Phase 1: CONNECT — handshake, verify account/instrument
Phase 2: HISTORY — receive all history bars, save to disk, wait for HISTORY_DONE
Phase 3: WARMUP — compute features on history, get SFE/aggregator ready
Phase 4: LIVE SYNC — process real-time bars, measure latency, log features

Usage:
    python live/diagnostic_run.py                 # all phases
    python live/diagnostic_run.py --phase 2       # stop after history
    python live/diagnostic_run.py --max-live 1000 # limit live bars
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import asyncio
import argparse
import logging
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger('diag')

from live.config import LiveConfig
from live.nt8_client import NT8Client
from live.protocol import MsgType
from training.aggregator import Aggregator
from core.statistical_field_engine import StatisticalFieldEngine
from training.compute_79d import compute_79d_from_aggregator, SFE_MIN_BARS
from core.features_79d import FEATURE_NAMES_79D, N_FEATURES


async def run_diagnostic(config, max_phase=4, max_live_bars=50000):
    """Single connection, 4 sequential phases."""

    client = NT8Client(config)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: CONNECT
    # ═══════════════════════════════════════════════════════════════════
    logger.info('=' * 60)
    logger.info('PHASE 1: CONNECT')
    logger.info('=' * 60)

    connected = await client.connect()
    if not connected:
        logger.error(f'Failed to connect to {config.nt8_host}:{config.nt8_port}')
        return

    # Read first few messages for handshake
    handshake = {}
    for _ in range(10):
        try:
            msg = await asyncio.wait_for(client.inbound.get(), timeout=10.0)
        except asyncio.TimeoutError:
            break

        msg_type = msg.get('type', '')
        if msg_type == 'CONNECTED':
            handshake = msg
            logger.info(f'  Account:    {msg.get("account")}')
            logger.info(f'  Instrument: {msg.get("instrument")}')
            logger.info(f'  Period:     {msg.get("primary_period_s")}s')
            logger.info(f'  Version:    {msg.get("version")}')
        elif msg_type == 'POSITION':
            logger.info(f'  Position:   qty={msg.get("qty")} avg={msg.get("avg_price")}')
        elif msg_type == 'ACCOUNT_UPDATE':
            logger.info(f'  Cash:       ${msg.get("cash_value"):,.0f}')
        elif msg_type == 'BAR':
            # First bar arrived — put it back conceptually, we'll process in phase 2
            # Actually we can't put it back in the queue, so we'll track it
            first_bar = msg
            break

    logger.info('Phase 1 PASS')
    if max_phase <= 1:
        await client.disconnect()
        return

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: HISTORY DUMP
    # ═══════════════════════════════════════════════════════════════════
    logger.info('')
    logger.info('=' * 60)
    logger.info('PHASE 2: HISTORY DUMP')
    logger.info('=' * 60)

    agg = Aggregator(history_limit=2000)
    history_bars = []
    history_done = False
    bar_count = 0

    # Process the first bar we already received
    if 'first_bar' in dir() and first_bar:
        bar = _extract_bar(first_bar)
        agg.feed(bar)
        history_bars.append(bar)
        bar_count = 1

    t0 = time.perf_counter()
    while not history_done:
        try:
            msg = await asyncio.wait_for(client.inbound.get(), timeout=120.0)
        except asyncio.TimeoutError:
            logger.warning('Timeout waiting for history bars (120s)')
            break

        if msg.get('type') == MsgType.BAR:
            bar = _extract_bar(msg)
            agg.feed(bar)
            history_bars.append(bar)
            bar_count += 1

            if bar_count % 2000 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(f'  {bar_count:,} bars ({elapsed:.1f}s) | '
                            f'1m={agg.get_bar_count("1m")} 5m={agg.get_bar_count("5m")} '
                            f'1h={agg.get_bar_count("1h")}')

        elif msg.get('type') == MsgType.HISTORY_DONE:
            history_done = True

    elapsed = time.perf_counter() - t0
    ts_first = history_bars[0]['timestamp'] if history_bars else 0
    ts_last = history_bars[-1]['timestamp'] if history_bars else 0

    logger.info(f'  History: {bar_count:,} bars in {elapsed:.1f}s')
    logger.info(f'  Range: {_ts_str(ts_first)} to {_ts_str(ts_last)}')
    logger.info(f'  TF bars: 1m={agg.get_bar_count("1m")} 5m={agg.get_bar_count("5m")} '
                f'15m={agg.get_bar_count("15m")} 1h={agg.get_bar_count("1h")}')

    # Save history bars as ATLAS_NT8 parquets — DELTA only (skip existing days)
    atlas_dir = 'DATA/ATLAS_NT8/5s'
    os.makedirs(atlas_dir, exist_ok=True)

    existing_days = set()
    if os.path.exists(atlas_dir):
        existing_days = {f.replace('.parquet', '') for f in os.listdir(atlas_dir) if f.endswith('.parquet')}

    hist_df = pd.DataFrame(history_bars)
    hist_df['day'] = pd.to_datetime(hist_df['timestamp'], unit='s').dt.strftime('%Y_%m_%d')
    all_days = sorted(hist_df['day'].unique())
    new_days = [d for d in all_days if d not in existing_days]

    logger.info(f'  Total days in history: {len(all_days)} | Already have: {len(existing_days)} | New: {len(new_days)}')

    if not new_days:
        logger.info(f'  No new days to save — ATLAS_NT8 up to date')
    else:
        for day in new_days:
            day_df = hist_df[hist_df['day'] == day]
            out = day_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            out['timestamp'] = out['timestamp'].astype(np.int64)
            out = out.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last').reset_index(drop=True)
            out.to_parquet(f'{atlas_dir}/{day}.parquet', index=False)

        # Aggregate new days to higher TFs
        for tf_name, tf_secs in [('15s', 15), ('1m', 60), ('5m', 300), ('15m', 900), ('1h', 3600), ('1D', 86400)]:
            tf_dir = f'DATA/ATLAS_NT8/{tf_name}'
            os.makedirs(tf_dir, exist_ok=True)
            for day in new_days:
                day_df = hist_df[hist_df['day'] == day]
                raw = day_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                raw['bucket'] = (raw['timestamp'] // tf_secs) * tf_secs
                agg_df = raw.groupby('bucket').agg({
                    'timestamp': 'first', 'open': 'first', 'high': 'max',
                    'low': 'min', 'close': 'last', 'volume': 'sum',
                }).reset_index(drop=True)
                agg_df['timestamp'] = agg_df['timestamp'].astype(np.int64)
                agg_df.to_parquet(f'{tf_dir}/{day}.parquet', index=False)

        logger.info(f'  Saved: {len(new_days)} new days ({new_days[0]} to {new_days[-1]})')
        logger.info(f'  Aggregated: 15s, 1m, 5m, 15m, 1h, 1D')

    logger.info('Phase 2 PASS')
    if max_phase <= 2:
        await client.disconnect()
        return

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: WARMUP (compute features on history)
    # ═══════════════════════════════════════════════════════════════════
    logger.info('')
    logger.info('=' * 60)
    logger.info('PHASE 3: WARMUP (features on history)')
    logger.info('=' * 60)

    sfe = StatisticalFieldEngine()
    prev_vel = {}

    # Compute 79D on current aggregator state
    feat, prev_vel, states_by_tf, _ = compute_79d_from_aggregator(
        agg, sfe, prev_vel, ts_last)

    if feat is not None:
        z = feat[12]   # 1m z
        vr = feat[14]  # 1m vr
        tfs = list(states_by_tf.keys())
        logger.info(f'  Last feature: z={z:.2f} vr={vr:.2f} TFs={tfs}')
        logger.info(f'  Feature vector: {N_FEATURES} dims')
        n_1m = agg.get_bar_count('1m')
        logger.info(f'  SFE ready: {n_1m} 1m bars (need {SFE_MIN_BARS})')
        if n_1m >= SFE_MIN_BARS:
            logger.info('  WARMED UP')
        else:
            logger.warning(f'  NOT WARMED UP — need {SFE_MIN_BARS - n_1m} more 1m bars')
    else:
        logger.warning('  Features returned None — not enough data')

    logger.info('Phase 3 PASS')
    if max_phase <= 3:
        await client.disconnect()
        return

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: LIVE SYNC (gaps, latency, parity)
    # ═══════════════════════════════════════════════════════════════════
    logger.info('')
    logger.info('=' * 60)
    logger.info('PHASE 4: LIVE SYNC')
    logger.info('=' * 60)

    pending_5s = None

    def on_bar_close(tf, bar):
        nonlocal pending_5s
        if tf == '5s':
            pending_5s = bar

    agg.on_bar_close = on_bar_close

    live_count = 0
    feat_count = 0
    latencies = []
    last_ts = 0
    last_recv = time.perf_counter()

    # Gap tracking
    gaps = []           # (ts, gap_s, type)
    duplicates = 0
    out_of_order = 0
    BAR_PERIOD = 5      # expected 5s between bars
    SESSION_GAP = 3600  # 1h+ = normal session break
    WARN_GAP = 30       # 30s+ = suspicious gap

    # Open ledger
    os.makedirs('reports/live', exist_ok=True)
    ledger_path = 'reports/live/diagnostic_ledger.csv'
    ledger = open(ledger_path, 'w')
    cols = ['timestamp', 'ts_str', 'price', 'recv_delay_ms', 'feat_ms',
            'gap_s', 'gap_type',
            'z', 'vr', 'vel', 'n_tfs'] + list(FEATURE_NAMES_79D)
    ledger.write(','.join(cols) + '\n')

    logger.info(f'  Listening for live bars... (max {max_live_bars})')
    logger.info(f'  Ledger: {ledger_path}')
    logger.info(f'  Gap thresholds: warn={WARN_GAP}s, session={SESSION_GAP}s')

    while live_count < max_live_bars:
        try:
            recv_t = time.perf_counter()
            msg = await asyncio.wait_for(client.inbound.get(), timeout=30.0)
        except asyncio.TimeoutError:
            if live_count > 0:
                logger.info(f'  30s timeout after {live_count} live bars — playback ended or disconnect')
            break

        if msg.get('type') != MsgType.BAR:
            continue

        bar = _extract_bar(msg)
        bar_ts = bar['timestamp']

        # ── Gap detection ──
        gap_s = 0
        gap_type = ''
        if last_ts > 0:
            gap_s = bar_ts - last_ts

            if gap_s == 0:
                duplicates += 1
                gap_type = 'DUPLICATE'
            elif gap_s < 0:
                out_of_order += 1
                gap_type = 'OUT_OF_ORDER'
            elif gap_s > SESSION_GAP:
                gap_type = 'SESSION_BREAK'
                gaps.append((bar_ts, gap_s, gap_type))
                logger.info(f'  SESSION BREAK: {gap_s:.0f}s gap at {_ts_str(bar_ts)}')
            elif gap_s > WARN_GAP:
                gap_type = 'GAP'
                gaps.append((bar_ts, gap_s, gap_type))
                logger.warning(f'  GAP: {gap_s:.0f}s at {_ts_str(bar_ts)} (expected {BAR_PERIOD}s)')
            elif gap_s != BAR_PERIOD:
                gap_type = f'SKIP_{int(gap_s)}s'

        # ── Receive delay (wall clock vs bar timestamp) ──
        recv_delay = (time.time() - bar_ts) * 1000 if bar_ts > 1e9 else 0
        inter_bar = (recv_t - last_recv) * 1000  # ms between receiving bars

        last_ts = bar_ts
        last_recv = recv_t

        # ── Feed aggregator ──
        pending_5s = None
        agg.feed(bar)
        live_count += 1

        if pending_5s is None:
            continue

        # ── Compute features ──
        feat_t0 = time.perf_counter()
        feat, prev_vel, states_by_tf, _ = compute_79d_from_aggregator(
            agg, sfe, prev_vel, bar['timestamp'])
        feat_ms = (time.perf_counter() - feat_t0) * 1000

        if feat is None:
            continue

        feat_count += 1
        z = feat[12]
        vr = feat[14]
        vel = feat[15]
        n_tfs = len(states_by_tf)
        latencies.append(feat_ms)

        # ── Write ledger ──
        row = [f'{bar_ts:.0f}', _ts_str(bar_ts), f'{bar["close"]:.2f}',
               f'{recv_delay:.0f}', f'{feat_ms:.1f}',
               f'{gap_s:.0f}', gap_type,
               f'{z:.4f}', f'{vr:.4f}', f'{vel:.2f}', str(n_tfs)]
        row += [f'{feat[i]:.6f}' for i in range(N_FEATURES)]
        ledger.write(','.join(row) + '\n')

        # ── Progress ──
        if feat_count <= 3 or feat_count % 200 == 0:
            logger.info(f'  LIVE {feat_count}: z={z:>+5.2f} vr={vr:.2f} '
                        f'price={bar["close"]:.2f} feat={feat_ms:.0f}ms '
                        f'{_ts_str(bar_ts)}')

        if feat_count % 1000 == 0:
            ledger.flush()

    ledger.flush()
    ledger.close()

    # ── SYNC REPORT ──
    logger.info('')
    logger.info('  SYNC REPORT:')
    logger.info(f'    Live bars:      {live_count:,}')
    logger.info(f'    Features:       {feat_count:,}')
    logger.info(f'    Duplicates:     {duplicates}')
    logger.info(f'    Out of order:   {out_of_order}')
    logger.info(f'    Gaps (>{WARN_GAP}s):  {len([g for g in gaps if g[2] == "GAP"])}')
    logger.info(f'    Session breaks: {len([g for g in gaps if g[2] == "SESSION_BREAK"])}')

    if latencies:
        lat = np.array(latencies)
        logger.info(f'    Feat latency:   avg={lat.mean():.0f}ms  p50={np.median(lat):.0f}ms  '
                    f'p99={np.percentile(lat, 99):.0f}ms  max={lat.max():.0f}ms')

    if gaps:
        logger.info(f'    Gap details:')
        for ts, gap_s, gtype in gaps[:20]:
            logger.info(f'      {_ts_str(ts)}: {gap_s:.0f}s ({gtype})')

    logger.info(f'    Ledger: {ledger_path}')
    logger.info('  Phase 4 DONE')
    await client.disconnect()


def _extract_bar(msg):
    return {
        'timestamp': msg.get('timestamp', 0),
        'open': msg.get('open', 0),
        'high': msg.get('high', 0),
        'low': msg.get('low', 0),
        'close': msg.get('close', 0),
        'volume': msg.get('volume', 0),
    }


def _ts_str(ts):
    from datetime import datetime, timezone
    if ts < 1e9:
        return '?'
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument('--max-live', type=int, default=50000)
    args = parser.parse_args()

    config = LiveConfig()
    logger.info(f'Config: {config.nt8_host}:{config.nt8_port} '
                f'account={config.account} instrument={config.instrument}')

    asyncio.run(run_diagnostic(config, args.phase, args.max_live))


if __name__ == '__main__':
    main()
