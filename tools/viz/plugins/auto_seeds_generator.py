"""
Auto Seeds Generator Plugin
Dynamically computes ZigZag pivots and generates seeds based on the currently loaded data in VizEngine.
Provides colored zones and size analysis.
"""
import argparse
import pandas as pd
import numpy as np

from tools.viz.core.plugin import VizPlugin

class AutoSeedsGeneratorPlugin(VizPlugin):
    
    def __init__(self, args):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--min-reversal', type=float, default=30.0, help='Minimum reversal in ticks (default 30)')
        parser.add_argument('--min-bars', type=int, default=5, help='Minimum bars duration (default 5)')
        parser.add_argument('--tick-size', type=float, default=0.25, help='Tick size (default 0.25)')
        self.args = parser.parse_args(args)
        
        self.seeds = []
        self.total_pnl = 0.0

    def _compute_zigzag(self):
        if self.engine.closes is None or len(self.engine.closes) < 2:
            self.seeds = []
            return
            
        closes = self.engine.closes
        pivots = [0]
        direction = None
        last_high = closes[0]
        last_low = closes[0]
        last_high_i = 0
        last_low_i = 0
        
        min_rev = self.args.min_reversal
        min_bars = self.args.min_bars
        tick = self.args.tick_size

        for i in range(1, len(closes)):
            if closes[i] > last_high:
                last_high = closes[i]
                last_high_i = i
            if closes[i] < last_low:
                last_low = closes[i]
                last_low_i = i

            if direction is None:
                if (last_high - last_low) / tick >= min_rev:
                    if last_high_i > last_low_i:
                        direction = 'UP'
                        pivots.append(last_low_i)
                    else:
                        direction = 'DOWN'
                        pivots.append(last_high_i)
            elif direction == 'UP':
                drop = (last_high - closes[i]) / tick
                if drop >= min_rev and (i - last_high_i) >= min_bars:
                    pivots.append(last_high_i)
                    direction = 'DOWN'
                    last_low = closes[last_high_i]
                    last_low_i = last_high_i
            elif direction == 'DOWN':
                rise = (closes[i] - last_low) / tick
                if rise >= min_rev and (i - last_low_i) >= min_bars:
                    pivots.append(last_low_i)
                    direction = 'UP'
                    last_high = closes[last_low_i]
                    last_high_i = last_low_i

        seeds = []
        for j in range(len(pivots) - 1):
            si = pivots[j]
            ei = pivots[j + 1]
            if ei <= si:
                continue
            d = 'LONG' if closes[ei] > closes[si] else 'SHORT'
            change = abs(closes[ei] - closes[si]) / tick
            seeds.append({
                'start_idx': si,
                'end_idx': ei,
                'start_dt': self.engine.dt.iloc[si],
                'end_dt': self.engine.dt.iloc[ei],
                'direction': d,
                'entry_price': closes[si],
                'exit_price': closes[ei],
                'change_ticks': change,
                'bars': ei - si,
            })
            
        self.seeds = seeds
        self.total_pnl = sum(s['change_ticks'] for s in self.seeds) * 0.50

    def draw(self, ax, time_range, patches_list):
        # Always recompute zigzag over the CURRENT engine data (supports dynamic zooming)
        self._compute_zigzag()
        
        if not self.seeds:
            return
            
        for s in self.seeds:
            color = '#2ecc71' if s['direction'] == 'LONG' else '#e74c3c'
            alpha = 0.2
            
            # Draw shaded zone
            p = ax.axvspan(s['start_dt'], s['end_dt'], alpha=alpha, color=color, zorder=0)
            patches_list.append(p)
            
            # Entry marker
            marker = '^' if s['direction'] == 'LONG' else 'v'
            m = ax.scatter([s['start_dt']], [s['entry_price']], marker=marker, color=color,
                           s=150, zorder=5, edgecolors='black', linewidths=1)
            patches_list.append(m)
            
            # PnL label at exit
            txt = ax.annotate(f"${s['change_ticks']*0.50:+.0f}",
                              xy=(s['end_dt'], s['exit_price']),
                              fontsize=8, color=color, fontweight='bold', ha='left',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor=color))
            patches_list.append(txt)

    def get_title_stats(self) -> str:
        return f"Auto Seeds (Rev: {self.args.min_reversal}t, Bars: {self.args.min_bars})  |  Count: {len(self.seeds)}  |  PnL: ${self.total_pnl:+,.0f}"

def get_plugin(unknown_args):
    return AutoSeedsGeneratorPlugin(unknown_args)
