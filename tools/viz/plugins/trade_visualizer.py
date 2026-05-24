"""
Trade Visualizer Plugin
Draws trades and MFE peak markers on top of the VizEngine's interactive chart.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from tools.viz.core.plugin import VizPlugin

class TradeVisualizerPlugin(VizPlugin):
    
    def __init__(self, args):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--trade-log', default='checkpoints/oracle_trade_log.csv')
        self.args = parser.parse_args(args)
        
        self.trades = None
        self.stats = {}

    def setup(self, engine, **kwargs):
        super().setup(engine, **kwargs)
        print(f"[Plugin] Loading trades from {self.args.trade_log}")
        try:
            self.trades = pd.read_csv(self.args.trade_log)
            # Filter valid times
            self.trades = self.trades.dropna(subset=['entry_time', 'exit_time'])
            
            self.trades['entry_dt'] = pd.to_datetime(self.trades['entry_time'], unit='s', utc=True)
            self.trades['exit_dt'] = pd.to_datetime(self.trades['exit_time'], unit='s', utc=True)
            
            # Precalculate wins
            if 'actual_pnl' in self.trades.columns:
                self.trades['is_win'] = self.trades['actual_pnl'] > 0
            else:
                self.trades['is_win'] = True # fallback
        except Exception as e:
            print(f"[Plugin] Failed to load trades: {e}")
            self.trades = pd.DataFrame()

    def draw(self, ax, time_range, patches_list):
        if self.trades is None or self.trades.empty:
            return
            
        # time_range is in matplotlib datenum format, convert engine's datetimes to this
        # Actually time_range is what get_xlim() returns. Let's filter trades that intersect.
        # Alternatively we can filter by the engine's current day date bounds.
        day_start = pd.Timestamp(self.engine.dt.iloc[0]).tz_localize('UTC')
        day_end = pd.Timestamp(self.engine.dt.iloc[-1]).tz_localize('UTC')
        
        # Add 1 hour buffer to catch overnight holds
        mask = (self.trades['entry_dt'] >= day_start - pd.Timedelta(hours=1)) & \
               (self.trades['entry_dt'] <= day_end + pd.Timedelta(hours=1))
        
        day_trades = self.trades[mask]
        
        n_trades = len(day_trades)
        wins = day_trades['is_win'].sum()
        pnl = day_trades['actual_pnl'].sum() if 'actual_pnl' in day_trades.columns else 0.0
        
        self.stats = {'n': n_trades, 'wins': wins, 'pnl': pnl}
        
        # Draw connecting lines and markers
        for _, t in day_trades.iterrows():
            is_win = t['is_win']
            is_long = t.get('direction', 'LONG') == 'LONG'
            
            entry_dt = t['entry_dt'].tz_localize(None) # Match engine timezone-naive
            exit_dt = t['exit_dt'].tz_localize(None)
            
            entry_px = t.get('entry_price', 0)
            exit_px = t.get('exit_price', 0)
            
            # Connecting line
            lc = '#00cc66' if is_win else '#ff4444'
            line, = ax.plot([entry_dt, exit_dt], [entry_px, exit_px],
                            color=lc, linewidth=1.5, alpha=0.5, zorder=3)
            patches_list.append(line)
            
            # Entry marker
            mc = '#00ff88' if is_win else '#ff6666'
            marker = '^' if is_long else 'v'
            en_m = ax.scatter([entry_dt], [entry_px], marker=marker, color=mc, s=80, 
                              zorder=6, edgecolors='black', linewidths=0.5)
            patches_list.append(en_m)
            
            # Exit marker
            ex_mc = '#00ff88' if is_win else '#ff4444'
            ex_m = ax.scatter([exit_dt], [exit_px], marker='x', color=ex_mc, s=50, zorder=5)
            patches_list.append(ex_m)
            
            # MFE Peak highlighter
            mfe = t.get('oracle_mfe', 0)
            if pd.notnull(mfe) and float(mfe) > 0:
                peak_px = entry_px + float(mfe) if is_long else entry_px - float(mfe)
                mid_dt = entry_dt + (exit_dt - entry_dt) / 2
                mfe_m = ax.scatter([mid_dt], [peak_px], marker='D', color='#ffaa00',
                                   s=40, zorder=7, alpha=0.8, edgecolors='#aa7700')
                patches_list.append(mfe_m)

    def get_title_stats(self) -> str:
        n = self.stats.get('n', 0)
        wins = self.stats.get('wins', 0)
        pnl = self.stats.get('pnl', 0.0)
        wr = (wins / n * 100) if n > 0 else 0.0
        return f"Trades: {n}  |  WR: {wr:.1f}%  |  Day PnL: ${pnl:,.2f}"

def get_plugin(unknown_args):
    return TradeVisualizerPlugin(unknown_args)
