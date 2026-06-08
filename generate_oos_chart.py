"""
generate_oos_chart.py
Plots the OOS PnL + Duration distributions for the most recent walk-forward
segment, drawn STRICTLY from the raw per-trade parquet persisted by
training.rl_engine.curriculum_metrics.OOSDiagnosticsSuite.

There is NO reconstruction-from-log-stats path. The prior version fabricated
a Normal distribution from scraped log moments (np.random.normal at line 81
of the old file) which produced an artificially symmetric chart whose μ did
not even match Net/n — that path is deleted. If no raw parquet exists, this
script prints NO_DATA and exits non-zero. A loud failure beats a silent lie.

The Normal curve drawn on top of the real histogram is a REFERENCE only
(labelled as such), to make the distribution's skew + tail-fatness visually
obvious — it is not a fit to data, and no data is sampled from it.

Diagnostic-only: read-only consumer of the parquet. Touches no training
state, no stopping rule, no loss term.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OOS_DIAG_DIR = os.path.join('reports', 'oos_diagnostics')
LEGACY_JSON  = os.path.join('oos_trade_data.json')  # transitional fallback
OUT_PATH     = os.path.join('oos_chart.png')


# ─── Data loading — raw only ─────────────────────────────────────────────────

def _list_segment_parquets():
    """Return list of (segment_id, build_tag, path) for every per-seg parquet."""
    if not os.path.isdir(OOS_DIAG_DIR):
        return []
    out = []
    pat = re.compile(r'^seg(\d+)_(.+)\.parquet$')
    for fn in os.listdir(OOS_DIAG_DIR):
        m = pat.match(fn)
        if m:
            seg_id = int(m.group(1))
            build_tag = m.group(2)
            out.append((seg_id, build_tag, os.path.join(OOS_DIAG_DIR, fn)))
    out.sort(key=lambda r: r[0])
    return out


def load_segment(segment_id=None):
    """
    Load the raw per-trade DataFrame for a segment.

    If segment_id is None, picks the highest-numbered segment on disk.
    Returns None if no parquet is available (caller must NO_DATA-exit).

    Transitional fallback: if no per-segment parquet exists yet but the
    legacy oos_trade_data.json is present (old single-rolling-file dump),
    promote it to an in-memory DataFrame so the chart still renders during
    the transition. The legacy JSON only has pnls + durations, so MFE/MAE
    columns will be missing — flagged in the chart subtitle.
    """
    parquets = _list_segment_parquets()

    if parquets:
        if segment_id is None:
            seg_id, build_tag, path = parquets[-1]
        else:
            match = [p for p in parquets if p[0] == segment_id]
            if not match:
                return None
            seg_id, build_tag, path = match[0]
        df = pd.read_parquet(path)
        df.attrs['source'] = 'parquet'
        df.attrs['segment_id'] = seg_id
        df.attrs['build_tag'] = build_tag
        df.attrs['path'] = path
        return df

    # Transitional fallback — strictly RAW (no synthesis), but reduced columns
    if os.path.exists(LEGACY_JSON):
        try:
            with open(LEGACY_JSON, 'r') as f:
                d = json.load(f)
        except Exception:
            return None
        pnls = d.get('pnls') or []
        durs = d.get('durations') or []
        if len(pnls) < 10:
            return None
        df = pd.DataFrame({
            'pnl_net': np.asarray(pnls, dtype=np.float64),
            'duration_bars': np.asarray(durs, dtype=np.float64),
        })
        df.attrs['source'] = 'legacy-json'
        df.attrs['segment_id'] = d.get('segment', '?')
        df.attrs['build_tag'] = 'unknown'
        df.attrs['path'] = LEGACY_JSON
        return df

    return None


# ─── Metrics from RAW arrays only ────────────────────────────────────────────

def real_metrics(pnls: np.ndarray) -> dict:
    """All metrics derived from the actual per-trade array. No moments."""
    n = len(pnls)
    if n == 0:
        return {}
    net = float(np.sum(pnls))
    mu = float(np.mean(pnls))
    # Consistency assertion — the bug this fix replaces failed this check
    net_per_trade = net / n
    assert abs(mu - net_per_trade) < 1e-9, (
        f'mean(pnls)={mu:.9f} != Net/n={net_per_trade:.9f}'
    )
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gp = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gl = float(np.sum(np.abs(losses))) if len(losses) > 0 else 0.0
    pf = (gp / gl) if gl > 0 else (float('inf') if gp > 0 else 0.0)
    return {
        'n': n,
        'net': net,
        'mu': mu,
        'sigma': float(np.std(pnls, ddof=1)) if n > 1 else 0.0,
        'pf': pf,
        'skew': float(stats.skew(pnls)) if n >= 8 else float('nan'),
        'excess_kurtosis': float(stats.kurtosis(pnls, fisher=True)) if n >= 8 else float('nan'),
        'p01': float(np.percentile(pnls, 1)),
        'p99': float(np.percentile(pnls, 99)),
    }


# ─── Plotting from raw ───────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame):
    pnls = np.asarray(df['pnl_net'].values, dtype=np.float64)
    durs = np.asarray(df['duration_bars'].values, dtype=np.float64)
    m = real_metrics(pnls)

    seg_id = df.attrs.get('segment_id', '?')
    build_tag = df.attrs.get('build_tag', 'unknown')
    source = df.attrs.get('source', 'unknown')
    has_mfe = 'mfe_available' in df.columns
    is_dirty_build = 'dirty' in str(build_tag) or build_tag == 'unknown'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')

    # Title-bar provenance band (red if build is dirty/unknown — VOID warning)
    void_banner = ''
    if build_tag == 'unknown':
        void_banner = '  ⚠ BUILD UNKNOWN — treat as VOID'
    elif 'dirty' in str(build_tag):
        void_banner = '  ⚠ build is -dirty (uncommitted changes)'
    fig.suptitle(
        f'OOS Trade Diagnostics — Seg {seg_id}  ·  build={build_tag}'
        f'  ·  source={source}{void_banner}',
        color='#f1c40f' if void_banner else 'white',
        fontsize=11, y=0.995,
    )

    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#e0e0e0')
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── PnL panel — drawn from REAL data ──
    ax1 = axes[0]
    lo, hi = m['p01'], m['p99']
    clipped = np.clip(pnls, lo, hi)
    n_bins = min(80, max(30, len(clipped) // 50))
    _, bin_edges, patches = ax1.hist(
        clipped, bins=n_bins, density=True, edgecolor='none',
    )
    for patch, left in zip(patches, bin_edges[:-1]):
        # Color by sign of bin (no exit_reason coloring available yet —
        # would require trainer-side per-trade exit tag)
        patch.set_facecolor('#e74c3c' if left < 0 else '#2ecc71')
        patch.set_alpha(0.65)

    # Scratch / breakeven marker — the real distribution often spikes near 0
    ax1.axvline(0, color='white', lw=1, ls='--', alpha=0.5, label='scratch ($0)')
    ax1.axvline(m['mu'], color='#f39c12', lw=1, ls=':', alpha=0.7,
                label=f'real μ={m["mu"]:.3f}')

    # REFERENCE Normal overlay — explicitly NOT a fit, just a visual yardstick
    # for seeing skew/kurtosis vs Gaussian. No data is sampled from it.
    x = np.linspace(lo, hi, 400)
    ax1.plot(
        x, stats.norm.pdf(x, m['mu'], max(m['sigma'], 1e-6)),
        color='#7f8c8d', lw=1.5, ls='-', alpha=0.8,
        label='reference Normal (not a fit to shape)',
    )

    ax1.set_title(
        'PnL (real) — μ={mu:.3f}  σ={sig:.2f}  skew={sk:+.2f}  '
        'excess kurt={kt:+.2f}  PF={pf:.4f}'.format(
            mu=m['mu'], sig=m['sigma'], sk=m['skew'],
            kt=m['excess_kurtosis'], pf=m['pf'],
        ),
        color='white', fontsize=10, pad=10,
    )
    ax1.set_xlabel('PnL ($)', color='#aaa')
    ax1.set_ylabel('Density', color='#aaa')
    ax1.legend(facecolor='#0f3460', labelcolor='white', fontsize=8)
    ax1.text(
        0.98, 0.97,
        f'n={m["n"]:,}  Net=${m["net"]:,.0f}\n'
        f'Net/n={m["net"]/m["n"]:.3f}  (== plotted μ)',
        transform=ax1.transAxes, ha='right', va='top',
        color='#ccc', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', alpha=0.7),
    )

    # ── Duration panel ──
    ax2 = axes[1]
    dur_hi = float(np.percentile(durs, 99))
    dur_c = np.clip(durs, 0.0, dur_hi)
    n_bins2 = min(60, max(20, int(dur_hi - durs.min()) // 2 + 1))
    ax2.hist(dur_c, bins=n_bins2, density=True,
             color='#9b59b6', alpha=0.65, edgecolor='none')
    med = float(np.median(durs))
    mean_d = float(np.mean(durs))
    ax2.axvline(med, color='#1abc9c', lw=1, ls='-', alpha=0.8,
                label=f'median={med:.1f}')
    ax2.axvline(mean_d, color='#f39c12', lw=1, ls=':', alpha=0.8,
                label=f'mean={mean_d:.1f}')
    ax2.set_title(
        'Duration (real, bars) — median={med:.1f}  mean={mn:.1f}  max={mx:.0f}'.format(
            med=med, mn=mean_d, mx=float(durs.max()),
        ),
        color='white', fontsize=10, pad=10,
    )
    ax2.set_xlabel('Duration (bars)', color='#aaa')
    ax2.set_ylabel('Density', color='#aaa')
    ax2.legend(facecolor='#0f3460', labelcolor='white', fontsize=8)

    # Footer — what's still missing from the schema (pending trainer-side
    # capture). Tells the reader to expect richer plots after that fix.
    pending = []
    if not has_mfe:
        pending.append('MFE/MAE')
    if 'exit_reason' not in df.columns:
        pending.append('exit_reason (no tail color-coding)')
    if 'entry_bar' not in df.columns:
        pending.append('entry_bar/exit_bar')
    if 'direction' not in df.columns:
        pending.append('direction')
    if pending:
        fig.text(
            0.5, 0.005,
            'Schema v1 — pending trainer-side capture: ' + ', '.join(pending),
            ha='center', color='#aaa', fontsize=8, style='italic',
        )

    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'OK seg={seg_id} build={build_tag} source={source} '
          f'n={m["n"]} mu={m["mu"]:.3f} pf={m["pf"]:.4f} '
          f'skew={m["skew"]:+.2f} excess_kurt={m["excess_kurtosis"]:+.2f}')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    seg_arg = None
    if len(sys.argv) > 1:
        try:
            seg_arg = int(sys.argv[1])
        except ValueError:
            pass

    df = load_segment(segment_id=seg_arg)
    if df is None or len(df) == 0:
        print('NO_DATA — no raw per-trade parquet found under '
              f'{OOS_DIAG_DIR} and no usable legacy {LEGACY_JSON}. '
              'Reconstruction-from-log-stats path was removed; this script '
              'now refuses to fabricate data.', file=sys.stderr)
        sys.exit(2)

    build_chart(df)


if __name__ == '__main__':
    main()
