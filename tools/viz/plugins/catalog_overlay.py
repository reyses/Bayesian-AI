import pandas as pd
import numpy as np
import os
import glob
import json
from tools.viz.core.plugin import VizPlugin

class CatalogOverlayPlugin(VizPlugin):
    def __init__(self, args):
        super().__init__()
        self.horizons = None
        self.labels = None

    def setup(self, engine, **kwargs):
        super().setup(engine, **kwargs)
        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        HORIZONS = os.path.join(ROOT, "research", "nt8_catalog", "reports", "fps_horizons.parquet")
        
        print(f"[CatalogOverlay] Loading {HORIZONS}")
        if os.path.exists(HORIZONS):
            df = pd.read_parquet(HORIZONS, columns=['doss', 'entry_ts', 'is_long'])
            df.loc[df['doss'] == 'ORB-02', 'entry_ts'] += 1800
            df = df[~df['doss'].isin(['SEASON-12', 'RENKO-24'])]
            df['datetime'] = pd.to_datetime(df['entry_ts'], unit='s', utc=True)
            self.horizons = df
        else:
            print("[CatalogOverlay] fps_horizons.parquet not found!")

        LABELS_DIR = os.path.join(ROOT, "DATA", "ai_cusp_picks")
        files = glob.glob(os.path.join(LABELS_DIR, "ai_picks_*_multi.json"))
        trades = []
        for f in files:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if 'trades' in data:
                    trades.extend(data['trades'])
        
        if trades:
            df_labels = pd.DataFrame(trades)
            df_labels['datetime'] = pd.to_datetime(df_labels['entry_ts'], unit='s', utc=True)
            self.labels = df_labels
            print(f"[CatalogOverlay] Loaded {len(self.labels)} golden labels")
        else:
            self.labels = None

    def draw(self, ax, time_range, patches_list):
        if self.horizons is None: return

        day_start = pd.Timestamp(self.engine.dt.iloc[0]).tz_localize('UTC')
        day_end = pd.Timestamp(self.engine.dt.iloc[-1]).tz_localize('UTC')

        mask_h = (self.horizons['datetime'] >= day_start) & (self.horizons['datetime'] <= day_end)
        day_h = self.horizons[mask_h]

        if self.labels is not None:
            mask_l = (self.labels['datetime'] >= day_start) & (self.labels['datetime'] <= day_end)
            day_l = self.labels[mask_l]
        else:
            day_l = pd.DataFrame()

        engine_dt_utc = self.engine.dt.dt.tz_localize('UTC')

        # 1. Plot Golden Labels (Large Gold Stars)
        if not day_l.empty:
            for _, row in day_l.iterrows():
                idx = np.argmin(np.abs(engine_dt_utc - row['datetime']))
                px = self.engine.closes[idx]
                dt_naive = row['datetime'].tz_localize(None)
                
                m = ax.scatter([dt_naive], [px], marker='*', color='gold', s=300, zorder=10, edgecolors='black')
                patches_list.append(m)

        # 2. Plot Catalog Entries
        if not day_h.empty:
            # Color map for dossiers
            dossiers = sorted(day_h['doss'].unique())
            cmap = __import__('matplotlib').pyplot.get_cmap('tab20')
            colors = {d: cmap(i % 20) for i, d in enumerate(dossiers)}
            
            for _, row in day_h.iterrows():
                idx = np.argmin(np.abs(engine_dt_utc - row['datetime']))
                px = self.engine.closes[idx]
                dt_naive = row['datetime'].tz_localize(None)
                
                is_long = row.get('is_long', True)
                # Triangle up for long, down for short
                marker = '^' if is_long else 'v'
                
                m = ax.scatter([dt_naive], [px], marker=marker, color=colors[row['doss']], s=60, zorder=5, alpha=0.7, edgecolors='white', linewidths=0.5)
                patches_list.append(m)

    def get_title_stats(self) -> str:
        return "Catalog Events vs Golden Labels"

def get_plugin(args):
    return CatalogOverlayPlugin(args)
