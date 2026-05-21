"""L5 Hybrid Sidecar — DEPRECATED 2026-05-18.

Status: research artifact / reference only. Do NOT run in production.

Reason for deprecation: this sidecar duplicates ~80% of `live/engine_v2.py`'s
existing infrastructure — pending-order tracking, fill reconciliation, NT8
transport, position state, mock-bridge support are all already implemented
in the engine_v2 + OrderManager + nt8_client + Ledger stack.

Correct path (see docs/L5_HYBRID_PIPELINE_SPEC.md):
  1. Build `training/live_feature_engine_v2.py` as a subclass of
     `LiveFeatureEngine` adding `get_v2_vector(ts)` (on-demand V2
     features, not streaming).
  2. Build `live/l5_decider.py` with same `evaluate(state) -> Batch`
     interface as `BlendedEngine` — runs zigzag/R-trigger/B7/B9 logic.
  3. Swap `self._engine = BlendedEngine(...)` for `L5Decider(...)` in
     `live/engine_v2.py` behind a `LiveConfig.engine_mode` flag.

The zigzag state machine and B9-at-K=5 sweep logic in this file are
worth lifting into `live/l5_decider.py`. Everything else (sidecar
protocol, IPC, threading) should be discarded.

Companion deprecation: `docs/nt8/ZigzagRunnerHybrid_v1.0.0-RC.cs`.

──────────────────────── ORIGINAL DOCSTRING ────────────────────────

L5 Hybrid Sidecar — applies B7 (entry) + B9 (during-trade) + B10 (day-level)
sizing on behalf of the NT8 ZigzagRunnerHybrid strategy.

PROTOCOL (JSON over TCP, length-prefixed per BayesianBridge.cs v7.0.0 pattern):

Wire format: [4 bytes uint32 big-endian payload length][N bytes UTF-8 JSON]

Incoming messages from NT8:
  DAY_OPEN       {day: "2026_05_18"}
    -> Server computes B10 day-multiplier from cross-day features.
    -> Caches it for subsequent ENTRY_QUERY calls this day.

  ENTRY_QUERY    {day, entry_ts, entry_price, leg_dir, position_id, v2_features: {...}}
    -> Server computes B7 size from v2_features at R-trigger fire bar.
    -> Multiplies by cached B10 day-mult.
    -> Returns ENTRY_ACTION with combined entry size.

  SIZE_QUERY     {position_id, entry_ts, current_ts, current_price, leg_dir,
                  v2_features: {...}, pnl_pts_so_far}
    -> Server computes B9 size_factor from v2 + trajectory at K=5 (T+25s).
    -> Returns SIZE_ACTION: HOLD | REDUCE_50 | CUT | PYRAMID.

  POSITION_CLOSED  {position_id, exit_pnl_usd, exit_ts}
    -> Server logs the outcome for monitoring.

Outgoing messages to NT8:
  ENTRY_ACTION   {position_id, contracts, b7_size, b10_mult, combined_size, reason}
  SIZE_ACTION    {position_id, action, b9_pred_remaining, size_factor, reason}
  ACK            {ok: true/false, error: ?}

DEPLOYMENT (post v2.1.0 of BayesianHistoryDumper):
  1. Run this sidecar: `python -m live.L5_sidecar --port 5200`
  2. Apply ZigzagRunnerHybrid_v1.0.0-RC to MNQ chart in NT8
  3. Strategy sends JSON messages to localhost:5200, sidecar replies

USAGE FOR BACKTEST PARITY:
  python -m live.L5_sidecar --offline-mode --legs-csv path/to/legs.csv
  -> Replays leg list without TCP, writes decisions to CSV.
"""
from __future__ import annotations
import argparse
import json
import logging
import pickle
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# === Model paths (production artifacts) ===
B7_PKL = Path('reports/findings/regret_oracle/b7_leg_sizer.pkl')
B9_PKL = Path('reports/findings/regret_oracle/b9_remaining_amplitude_K5.pkl')
B10_HIGH_PKL = Path('reports/findings/regret_oracle/b10_vol_regime_high.pkl')
B10_LOW_PKL  = Path('reports/findings/regret_oracle/b10_vol_regime_low.pkl')

CROSS_DAY_FEATURES = Path('DATA/CROSS_DAY/cross_day_features.parquet')

# === B10 action thresholds (selected from IS WF — DO NOT TUNE ON OOS) ===
B10_THR_HIGH = 0.5
B10_THR_LOW  = 0.7
B10_BOOST = 1.3
B10_CAP   = 0.7

# === B10 feature columns ===
B10_FEATS = [
    'overnight_gap_pct', 'overnight_range_pct',
    'prior_day_range_pct', 'prior_day_c2c_pct',
    'vix_close_prior', 'vix_chg_prior',
    'dxy_close_prior', 'dxy_chg_prior',
    'is_fomc', 'is_cpi', 'is_nfp', 'is_opex',
    'days_since_fomc', 'days_to_next_fomc', 'dow',
]


# === Sizing rules ===

def b7_size(pred_amp_R: float) -> float:
    """gbm_ev: clip(max(pred_R - 1, 0), 0, 3). Returns multiplier on 1 contract."""
    return float(np.clip(max(pred_amp_R - 1.0, 0.0), 0.0, 3.0))


def b9_size_from_pred(pred_remaining_usd: float) -> tuple[float, str]:
    """B9 sizing rule + action label."""
    if pred_remaining_usd > 50:
        return 1.5, 'PYRAMID'
    if pred_remaining_usd > 10:
        return 1.0, 'HOLD'
    if pred_remaining_usd > -10:
        return 1.0, 'HOLD_UNCERTAIN'
    if pred_remaining_usd > -50:
        return 0.5, 'REDUCE_50'
    return 0.0, 'CUT'


def b10_day_mult(p_high: float, p_low: float) -> tuple[float, str]:
    """B10 day-level multiplier (mutually exclusive: high beats low)."""
    if p_high >= B10_THR_HIGH:
        return B10_BOOST, 'BOOST_HIGH_VOL'
    if p_low >= B10_THR_LOW:
        return B10_CAP, 'CAP_LOW_VOL'
    return 1.0, 'HOLD_NORMAL_VOL'


# === Sidecar engine ===

class L5Sidecar:
    """Holds production models + day-cache. Handles JSON queries from NT8."""

    def __init__(self, models_dir: Path = Path('reports/findings/regret_oracle')):
        self.log = logging.getLogger('L5Sidecar')
        self._load_models(models_dir)
        self._day_cache: dict[str, dict] = {}   # day -> {b10_mult, p_high, p_low, ...}
        self._positions: dict[str, dict] = {}   # position_id -> entry info

    def _load_models(self, models_dir: Path):
        self.log.info(f'Loading B7 from {B7_PKL}')
        with open(B7_PKL, 'rb') as f:
            self.b7 = pickle.load(f)
        self.log.info(f'  B7 v2_cols: {len(self.b7["v2_cols"])}')

        self.log.info(f'Loading B9 from {B9_PKL}')
        with open(B9_PKL, 'rb') as f:
            self.b9 = pickle.load(f)
        self.log.info(f'  B9 feat_cols: {len(self.b9["feat_cols"])}')

        self.log.info(f'Loading B10 from {B10_HIGH_PKL} + {B10_LOW_PKL}')
        with open(B10_HIGH_PKL, 'rb') as f:
            self.b10_high = pickle.load(f)
        with open(B10_LOW_PKL, 'rb') as f:
            self.b10_low = pickle.load(f)

        # Pre-load cross-day features for B10 lookups
        if CROSS_DAY_FEATURES.exists():
            self.cross_day = pd.read_parquet(CROSS_DAY_FEATURES).set_index('date_label')
            self.log.info(f'  cross_day_features: {len(self.cross_day)} days indexed')
        else:
            self.cross_day = pd.DataFrame()
            self.log.warning(f'  cross_day_features missing at {CROSS_DAY_FEATURES}')

    def handle_day_open(self, msg: dict) -> dict:
        """Compute B10 multiplier for the day."""
        day = msg['day']
        if day in self._day_cache:
            return {'ack': True, 'cached': True, **self._day_cache[day]}

        if day not in self.cross_day.index:
            self.log.warning(f'DAY_OPEN {day}: no cross_day_features row, default 1.0x')
            self._day_cache[day] = {
                'b10_mult': 1.0, 'p_high': 0.0, 'p_low': 0.0,
                'reason': 'NO_FEATURES_AVAILABLE',
            }
            return {'ack': True, **self._day_cache[day]}

        row = self.cross_day.loc[day]
        X = np.array([float(row.get(c, 0.0) or 0.0) for c in B10_FEATS],
                       dtype=np.float32).reshape(1, -1)
        p_high = float(self.b10_high['model'].predict_proba(X)[0, 1])
        p_low  = float(self.b10_low['model'].predict_proba(X)[0, 1])
        mult, reason = b10_day_mult(p_high, p_low)
        self._day_cache[day] = {
            'b10_mult': mult, 'p_high': p_high, 'p_low': p_low, 'reason': reason,
        }
        self.log.info(f'DAY_OPEN {day}: p_high={p_high:.3f} p_low={p_low:.3f} '
                       f'-> {mult}x ({reason})')
        return {'ack': True, **self._day_cache[day]}

    def handle_entry_query(self, msg: dict) -> dict:
        """Apply B7 sizing at R-trigger fire bar."""
        day = msg['day']
        position_id = msg['position_id']
        v2_features = msg.get('v2_features', {})

        # B10 mult — use cached if available, else compute
        if day not in self._day_cache:
            self.handle_day_open({'day': day})
        b10_mult = self._day_cache[day]['b10_mult']

        # B7 prediction
        X = np.array([float(v2_features.get(c, 0.0) or 0.0)
                       for c in self.b7['v2_cols']],
                       dtype=np.float32).reshape(1, -1)
        b7_pred_amp_R = float(self.b7['model'].predict(X)[0])
        b7_sz = b7_size(b7_pred_amp_R)

        combined = b7_sz * b10_mult
        # Convert to contracts (assume base 1 contract; round)
        contracts = max(0, int(round(combined)))

        # Cache entry context for later SIZE_QUERY
        self._positions[position_id] = {
            'day': day,
            'entry_ts': msg.get('entry_ts'),
            'entry_price': msg.get('entry_price'),
            'leg_dir': msg.get('leg_dir'),
            'b7_size': b7_sz,
            'b7_pred_amp_R': b7_pred_amp_R,
            'b10_mult': b10_mult,
            'combined_size': combined,
            'contracts': contracts,
        }

        self.log.info(f'ENTRY_QUERY {position_id} {day} {msg.get("leg_dir")}: '
                       f'b7_pred={b7_pred_amp_R:.2f}R -> b7_size={b7_sz:.2f}, '
                       f'b10_mult={b10_mult:.2f}x, combined={combined:.2f} '
                       f'-> {contracts} contracts')

        return {
            'ack': True,
            'position_id': position_id,
            'contracts': contracts,
            'b7_size': b7_sz,
            'b7_pred_amp_R': b7_pred_amp_R,
            'b10_mult': b10_mult,
            'combined_size': combined,
            'reason': 'B7_ENTRY_SIZING',
        }

    def handle_size_query(self, msg: dict) -> dict:
        """Apply B9 sizing at T+25s (K=5 bar)."""
        position_id = msg['position_id']
        v2_features = msg.get('v2_features', {})
        pnl_pts_so_far = float(msg.get('pnl_pts_so_far', 0.0))

        if position_id not in self._positions:
            self.log.warning(f'SIZE_QUERY {position_id}: position unknown, default HOLD')
            return {'ack': True, 'position_id': position_id,
                     'action': 'HOLD', 'size_factor': 1.0,
                     'reason': 'POSITION_NOT_TRACKED'}

        entry = self._positions[position_id]

        # B9 prediction
        # Construct feature vector matching b9['feat_cols'] from v2 + trajectory
        # The trajectory features (mae_pts_so_far, mfe_pts_so_far, pnl_pts_so_far,
        # pnl_usd_so_far, bars_since_entry, has_reached_R_against) must be
        # supplied by the NT8 caller in v2_features.
        X = np.array([float(v2_features.get(c, 0.0) or 0.0)
                       for c in self.b9['feat_cols']],
                       dtype=np.float32).reshape(1, -1)
        b9_pred = float(self.b9['model'].predict(X)[0])
        size_factor, action = b9_size_from_pred(b9_pred)

        self.log.info(f'SIZE_QUERY {position_id}: b9_pred={b9_pred:+.2f}$ '
                       f'pnl_pts_so_far={pnl_pts_so_far:+.2f} '
                       f'-> {action} (sf={size_factor})')

        return {
            'ack': True,
            'position_id': position_id,
            'action': action,
            'size_factor': size_factor,
            'b9_pred_remaining_usd': b9_pred,
            'reason': f'B9_K5_DECISION_{action}',
        }

    def handle_position_closed(self, msg: dict) -> dict:
        position_id = msg['position_id']
        exit_pnl = float(msg.get('exit_pnl_usd', 0.0))
        self.log.info(f'POSITION_CLOSED {position_id}: exit_pnl=${exit_pnl:+.2f}')
        # Could write to a monitoring CSV here
        return {'ack': True}

    def dispatch(self, msg: dict) -> dict:
        kind = msg.get('type', '').upper()
        if kind == 'DAY_OPEN':
            return self.handle_day_open(msg)
        elif kind == 'ENTRY_QUERY':
            return self.handle_entry_query(msg)
        elif kind == 'SIZE_QUERY':
            return self.handle_size_query(msg)
        elif kind == 'POSITION_CLOSED':
            return self.handle_position_closed(msg)
        else:
            return {'ack': False, 'error': f'unknown message type: {kind}'}


# === TCP server (length-prefixed JSON) ===

def serve_tcp(sidecar: L5Sidecar, host: str, port: int):
    log = logging.getLogger('TCP')
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    log.info(f'L5 sidecar listening on {host}:{port}')

    while True:
        client, addr = server.accept()
        log.info(f'connection from {addr}')
        try:
            while True:
                # Read 4-byte length prefix
                header = client.recv(4, socket.MSG_WAITALL)
                if not header or len(header) < 4:
                    break
                payload_len = struct.unpack('>I', header)[0]
                if payload_len <= 0 or payload_len > 1_000_000:
                    log.warning(f'bad payload length {payload_len}, closing')
                    break

                payload = client.recv(payload_len, socket.MSG_WAITALL)
                if len(payload) < payload_len:
                    log.warning(f'short payload, closing')
                    break

                try:
                    msg = json.loads(payload.decode('utf-8'))
                except Exception as e:
                    log.warning(f'bad JSON: {e}')
                    client.sendall(b'\x00\x00\x00\x10{"ack":false,"error":"bad_json"}')
                    continue

                response = sidecar.dispatch(msg)
                resp_bytes = json.dumps(response).encode('utf-8')
                resp_len = struct.pack('>I', len(resp_bytes))
                client.sendall(resp_len + resp_bytes)
        except Exception as e:
            log.exception(f'connection error: {e}')
        finally:
            client.close()
            log.info(f'closed {addr}')


# === Offline replay mode (for backtest parity) ===

def offline_replay(sidecar: L5Sidecar, legs_csv: Path, trajectory_parquet: Path,
                   out_csv: Path):
    """Replay a leg list through the sidecar without TCP. Useful for verifying
    Python-side decisions match the forward_pass_full_stack output."""
    log = logging.getLogger('Offline')
    legs = pd.read_csv(legs_csv)
    log.info(f'Legs: {len(legs)} across {legs["day"].nunique()} days')

    traj = pd.read_parquet(trajectory_parquet)
    traj_k5 = traj[traj['K'] == 5].set_index('leg_id')

    out_rows = []
    for day in sorted(legs['day'].unique()):
        sidecar.handle_day_open({'day': day})

    for idx, leg in legs.iterrows():
        # Build fake v2_features for B7 (we don't have them in the leg CSV
        # alone — would need to look up from truth dataset).
        # For offline replay, this just confirms protocol works; actual
        # backtest stays in forward_pass_full_stack.py.
        position_id = f'leg_{idx}'

        # Skip if no trajectory K=5 row
        if idx not in traj_k5.index:
            continue
        traj_row = traj_k5.loc[idx]

        # B7 needs the V2 entry features; we approximate from traj features here
        # since the offline replay is a protocol check, not a $/day evaluator
        b9_features = {c: float(traj_row.get(c, 0.0) or 0.0)
                        for c in sidecar.b9['feat_cols']}

        # SIZE_QUERY for B9
        size_resp = sidecar.handle_size_query({
            'position_id': position_id,
            'v2_features': b9_features,
            'pnl_pts_so_far': float(traj_row.get('pnl_pts_so_far', 0.0)),
        })

        out_rows.append({
            'leg_id': idx, 'day': leg['day'],
            'b9_action': size_resp['action'],
            'b9_size_factor': size_resp['size_factor'],
            'b9_pred_remaining_usd': size_resp.get('b9_pred_remaining_usd'),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    log.info(f'Wrote: {out_csv} ({len(out_df)} rows)')


# === Main ===

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5200,
                    help='TCP port (default 5200; BayesianBridge uses 5199)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--offline-replay', action='store_true',
                    help='Skip TCP; replay legs CSV instead')
    ap.add_argument('--legs-csv',
                    default='reports/findings/regret_oracle/oos_hardened_legs_full.csv')
    ap.add_argument('--trajectory',
                    default='reports/findings/regret_oracle/trade_trajectory_OOS_full.parquet')
    ap.add_argument('--offline-out', default='reports/findings/regret_oracle/l5_offline_replay.csv')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s %(name)s %(levelname)s %(message)s')

    sidecar = L5Sidecar()

    if args.offline_replay:
        offline_replay(sidecar, Path(args.legs_csv), Path(args.trajectory),
                        Path(args.offline_out))
    else:
        serve_tcp(sidecar, args.host, args.port)


if __name__ == '__main__':
    main()
