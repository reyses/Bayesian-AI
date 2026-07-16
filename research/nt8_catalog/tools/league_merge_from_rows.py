"""
Rebuild the FULL 12-stream league md from saved signal_rows parquets.

The 2026-07-15 disk-full crash split the league across two runs; row parquets for all
12 streams survived. This re-runs dossier_signal_pipeline.evaluate() on each saved
row set (identical data -> identical logistic; bootstrap CIs re-drawn) and writes one
complete reports/dossier_signal_league.md.
"""
import os, glob, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dossier_signal_pipeline as dsp

ORDER = ['ZIGZAG', 'ORB-02', 'SEASON-12', 'VWAP-03', 'OHLC-01', 'PIVOT-16',
         'ROUND-05', 'CROSS-11', 'VWMA-10', 'DOW-19', 'TUNNEL-20', 'ATR-09',
         'SAR-23', 'SQZ-04', 'RSI-06', 'MACD-07', 'SCALP-18', 'RENKO-24',
         'FIB-17', 'ZONE-21', 'VP-01', 'VA-13', 'HNS-22', 'CURVE', 'ADX08']

lblf = {os.path.basename(f)[9:19]: f
        for f in glob.glob(os.path.join(dsp.LBL, 'ai_picks_*_multi.json'))}
lines = ['# Dossier signal league — direction agreement with AI labels',
         '(train 2024, test 2025+26, day-block bootstrap CIs; baseline 0.50)\n']
for det in ORDER:
    p = os.path.join(dsp.REP, f'signal_rows_{det.replace("-", "")}.parquet')
    if not os.path.exists(p):
        lines.append(f'- **{det}**: no saved rows (see run log / skip list)')
        print(det, 'no saved rows')
        continue
    F = pd.read_parquet(p)[['ts', 'is_long', 'value', 'pivot_age_min',
                            'sig_with_leg', 'tod', 'day']]
    r = dsp.evaluate(det, F, lblf)
    if 'note' in r:
        lines.append(f'- **{det}**: N={r["n"]} — {r["note"]}')
        print(det, r['note'])
        continue
    t = r['ter']
    ts = ' | '.join(f'{b}: {t[b][1]:.2f} [{t[b][2]:.2f},{t[b][3]:.2f}] N={t[b][0]}' for b in t)
    print(f'{det:10} N={r["n"]:6} OOS-AUC {r["auc"]:.3f} base {r["base_te"]:.2f} || {ts}')
    lines.append(f'## {det}\n- N={r["n"]} (train {r["n_tr"]} / test {r["n_te"]}), '
                 f'OOS AUC **{r["auc"]:.3f}**, test base {r["base_te"]:.2f}\n'
                 f'- P-terciles: {ts}\n- coefs: {r["coefs"]}')
with open(os.path.join(dsp.REP, 'dossier_signal_league.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('wrote complete 12-stream dossier_signal_league.md')
