"""
DOJO mode for the VizEngine.

Visualises a dojo-forge EXIT-drill episode on the price chart: the entry, and
the teacher's per-frame HOLD/EXIT decisions (from the gate transcript) placed at
each frame's minute, with the committed EXIT and its reason highlighted. Cycle
episodes on the day with n/b; press r to RUN the qwen3 teacher LIVE on the
current episode — decisions stream onto the chart as the model commits each frame.

Run:
    python -m tools.viz.run --plugin dojo_replay --day 2025_01_21
Keys:  n / b  next / prev episode      r  run teacher live on this episode
Data:  packets  research/dojo_forge/reports/gen0/packets/<eid>.json
       replay   research/dojo_forge/gate_state/gen0/<eid>.transcript.jsonl
       live     research/dojo_forge/gate_state/viz_live/<eid>.transcript.jsonl
"""
import glob
import json
import os
import re
import subprocess
import threading

import numpy as np
import pandas as pd

from tools.viz.core.plugin import VizPlugin

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DOJO = os.path.join(REPO, 'research', 'dojo_forge')
PY = '/home/moi/miniforge3/envs/bayesian/bin/python'
_NV = '/home/moi/miniforge3/envs/bayesian/lib/python3.12/site-packages/nvidia'

HOLD_C, EXIT_C = '#2E7D32', '#C62828'
LONG_C, SHORT_C = '#1565C0', '#AD1457'
_PX = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*pts')


class DojoReplayPlugin(VizPlugin):
    def __init__(self, args):
        super().__init__()
        a = args or []
        self.packets_dir = _opt(a, '--packets-dir', os.path.join(DOJO, 'reports', 'gen0', 'packets'))
        self.replay_dir = _opt(a, '--transcript-dir', os.path.join(DOJO, 'gate_state', 'gen0'))
        self.live_dir = os.path.join(DOJO, 'gate_state', 'viz_live')
        self.model_blob = _opt(a, '--model-blob',
                               '/media/moi/WindowsCode/ollama/models/blobs/'
                               'sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e')
        self.epi_idx = 0
        self.episodes = []
        self.live = False
        self._live_eid = None
        self._timer = None
        self._cur_day = None
        self.blind = '--blind' in a      # reveal only up to the current frame
        self.cur_frame = 0               # revealed frame cursor (blind mode)

    # ---- episode discovery -------------------------------------------------
    def _day(self):
        return self.engine.days[self.engine.day_idx]

    def _refresh_episodes(self):
        day = self._day()
        if day == self._cur_day:
            return
        self._cur_day = day
        eids = []
        for p in sorted(glob.glob(os.path.join(self.packets_dir, f'{day}_*.json'))):
            eids.append(os.path.basename(p)[:-5])
        self.episodes = eids
        self.epi_idx = 0
        self.cur_frame = 0

    def _cur_eid(self):
        return self.episodes[self.epi_idx] if self.episodes else None

    # ---- data loading ------------------------------------------------------
    @staticmethod
    def _entry_ts(eid):
        return int(eid.split('_')[3])

    def _load_packet(self, eid):
        with open(os.path.join(self.packets_dir, f'{eid}.json'), encoding='utf-8') as f:
            return json.load(f)

    def _load_decisions(self, eid):
        """frame_int -> (decision, reason). Live dir wins if a live run exists."""
        out = {}
        for d in (self.live_dir, self.replay_dir):
            path = os.path.join(d, f'{eid}.transcript.jsonl')
            if not os.path.exists(path):
                continue
            try:
                for ln in open(path, encoding='utf-8'):
                    r = json.loads(ln)
                    if r.get('event') == 'commit':
                        out[int(r['frame'])] = (r.get('decision', '?'), r.get('reason', ''))
            except (ValueError, OSError):
                pass
            if out:
                return out
        return out

    # ---- price mapping -----------------------------------------------------
    def _xy(self, ts_epoch):
        """Chart (x_datetime, y_price) for an epoch, snapped to the price line."""
        ep = self.engine.dt.astype('int64') // 10 ** 9
        y = float(np.interp(ts_epoch, ep.values, self.engine.closes))
        return pd.to_datetime(ts_epoch, unit='s'), y

    # ---- draw --------------------------------------------------------------
    def draw(self, ax, ax_ind, time_range, patches):
        self._refresh_episodes()
        eid = self._cur_eid()
        if not eid:
            patches.append(ax.text(0.5, 0.5, f'No dojo episodes for {self._day()}',
                                   transform=ax.transAxes, ha='center', color='gray', fontsize=13))
            return
        pkt = self._load_packet(eid)
        meta = pkt.get('meta', {})
        frames = pkt['frames']
        nframes = len(frames)
        d = 1 if str(meta.get('direction', '')).upper().startswith('L') else -1
        dc = LONG_C if d > 0 else SHORT_C
        ent = self._entry_ts(eid)
        dec = self._load_decisions(eid)
        self.cur_frame = max(0, min(self.cur_frame, nframes - 1))
        cutoff = self.cur_frame if self.blind else nframes - 1   # last visible frame

        ex, ey = self._xy(ent)
        patches.append(ax.scatter([ex], [ey], marker='^' if d > 0 else 'v',
                                  s=180, color=dc, edgecolor='black', zorder=6))
        patches.append(ax.annotate(f'entry {meta.get("direction", "?")}', (ex, ey),
                                    textcoords='offset points', xytext=(0, 14 * d),
                                    ha='center', color=dc, fontsize=9, fontweight='bold'))

        # per-frame decision path + favorable-pts subplot (only up to cutoff)
        fav = []
        px, py = [ex], [ey]
        for fr in frames:
            fnum = int(fr['frame'])
            if fnum > cutoff:
                break
            fts = ent + fnum * 60
            x, y = self._xy(fts)
            px.append(x); py.append(y)
            m = _PX.search(fr.get('text', ''))
            fav.append((fnum, float(m.group(1)) if m else np.nan))
            if fnum in dec:
                deci, reason = dec[fnum]
                is_exit = deci.upper().startswith('EXIT')
                patches.append(ax.scatter([x], [y], marker='x' if is_exit else 'o',
                                          s=140 if is_exit else 42,
                                          color=EXIT_C if is_exit else HOLD_C,
                                          linewidths=2.5 if is_exit else 1.0, zorder=7))
                if is_exit:
                    patches.append(ax.axvline(x, color=EXIT_C, ls='--', lw=1.0, alpha=0.6, zorder=3))
                    patches.append(ax.annotate('EXIT: ' + reason[:70], (x, y),
                                               textcoords='offset points', xytext=(6, -18),
                                               fontsize=8, color=EXIT_C,
                                               bbox=dict(boxstyle='round', fc='#FFF3F3', ec=EXIT_C, alpha=0.9)))
        ln, = ax.plot(px, py, color=dc, lw=1.0, alpha=0.35, zorder=4)
        patches.append(ln)

        # BLIND: fog the future (everything after the current frame) + show the
        # exact frame text qwen is reading, so you see only what it sees.
        if self.blind:
            fog_x, _ = self._xy(ent + cutoff * 60)
            right = self.engine.dt.iloc[-1]
            patches.append(ax.axvspan(fog_x, right, facecolor='#FAFAFA',
                                      edgecolor='none', alpha=1.0, zorder=5))
            patches.append(ax.axvline(fog_x, color='#455A64', lw=1.2, alpha=0.8, zorder=6))
            ftext = frames[cutoff].get('text', '')
            lines = ftext.splitlines()
            shown = '\n'.join(lines[:20]) + ('\n  …' if len(lines) > 20 else '')
            patches.append(ax.text(0.008, 0.985, shown, transform=ax.transAxes,
                                   va='top', ha='left', family='monospace', fontsize=6.5,
                                   color='#263238', zorder=9,
                                   bbox=dict(boxstyle='round', fc='#FFFDE7', ec='#455A64', alpha=0.95)))
            patches.append(ax.text(0.5, 0.985, f'BLIND — frame {cutoff}/{nframes-1} '
                                   '(you see only what qwen sees)', transform=ax.transAxes,
                                   ha='center', va='top', fontsize=10, fontweight='bold',
                                   color='#455A64', zorder=9))

        if ax_ind is not None and fav:
            ax_ind.clear()
            ax_ind.grid(True, alpha=0.2); ax_ind.set_facecolor('#FAFAFA')
            fn = [f for f, _ in fav]; fv = [v for _, v in fav]
            ax_ind.plot(fn, fv, '-', color='#607D8B', lw=1.2)
            ax_ind.axhline(0, color='black', lw=0.8)
            for f, v in fav:
                if f in dec and not np.isnan(v):
                    is_exit = dec[f][0].upper().startswith('EXIT')
                    ax_ind.scatter([f], [v], marker='x' if is_exit else 'o',
                                   color=EXIT_C if is_exit else HOLD_C, s=60 if is_exit else 28, zorder=5)
            ax_ind.set_ylabel('fav pts', fontsize=9)
            ax_ind.set_xlabel('frame (min from entry)', fontsize=9)

    def get_title_stats(self):
        eid = self._cur_eid()
        if not eid:
            return 'DOJO — no episodes this day'
        dec = self._load_decisions(eid)
        served = len(dec)
        exitf = next((f for f in sorted(dec) if dec[f][0].upper().startswith('EXIT')), None)
        tag = 'LIVE ▶ ' if (self.live and self._live_eid == eid) else ''
        bl = f'BLIND f{self.cur_frame} · ' if self.blind else ''
        st = f'{tag}{bl}DOJO {self.epi_idx + 1}/{len(self.episodes)}  {eid}  |  ' \
             f'frames decided {served}  |  ' + \
             (f'EXIT@{exitf}' if exitf is not None else 'no EXIT yet') + \
             '   [n/b episode · f/v frame · x blind · r run-live]'
        return st

    # ---- interaction -------------------------------------------------------
    def on_key(self, event):
        k = (event.key or '').lower()
        if k in ('n', 'm') and self.episodes:
            self.epi_idx = (self.epi_idx + 1) % len(self.episodes)
            self.cur_frame = 0; self.engine.draw(); return True
        if k == 'b' and self.episodes:
            self.epi_idx = (self.epi_idx - 1) % len(self.episodes)
            self.cur_frame = 0; self.engine.draw(); return True
        if k == 'x':                                   # toggle blind fog
            self.blind = not self.blind; self.engine.draw(); return True
        if k == 'f':                                   # reveal next frame
            self.cur_frame += 1; self.engine.draw(); return True
        if k == 'v':                                   # step back a frame
            self.cur_frame = max(0, self.cur_frame - 1); self.engine.draw(); return True
        if k == 'r':
            self._run_live()
            return True
        return False

    def _run_live(self):
        eid = self._cur_eid()
        if not eid or self.live:
            return
        os.makedirs(self.live_dir, exist_ok=True)
        # clear any stale live transcript so the stream starts fresh
        old = os.path.join(self.live_dir, f'{eid}.transcript.jsonl')
        if os.path.exists(old):
            os.remove(old)
        self.live = True
        self._live_eid = eid
        env = dict(os.environ)
        env['DOJO_RUN_DIR'] = os.path.dirname(self.packets_dir)   # .../gen0 -> packets/
        env['LD_LIBRARY_PATH'] = ':'.join(
            [f'{_NV}/cuda_runtime/lib', f'{_NV}/cublas/lib', f'{_NV}/cudnn/lib',
             f'{_NV}/cuda_nvrtc/lib', env.get('LD_LIBRARY_PATH', '')])
        cmd = [PY, os.path.join(DOJO, 'pipeline', 'forge_harness.py'),
               '--episodes', eid, '--run-id', 'viz_live', '--model-blob', self.model_blob]

        def worker():
            try:
                subprocess.run(cmd, env=env, cwd=REPO,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                self.live = False

        threading.Thread(target=worker, daemon=True).start()
        # poll the transcript and redraw while the model works
        self._timer = self.engine.fig.canvas.new_timer(interval=1500)
        self._timer.add_callback(self._tick)
        self._timer.start()
        self.engine.draw()

    def _tick(self):
        # follow qwen: advance the blind reveal to the frame it's currently on
        if self._live_eid:
            path = os.path.join(self.live_dir, f'{self._live_eid}.transcript.jsonl')
            mx = -1
            try:
                for ln in open(path, encoding='utf-8'):
                    r = json.loads(ln)
                    if r.get('event') in ('serve', 'commit'):
                        mx = max(mx, int(r['frame']))
            except (ValueError, OSError):
                pass
            if mx >= 0:
                self.cur_frame = mx
        try:
            self.engine.draw()
        except Exception:
            pass
        if not self.live and self._timer is not None:
            self._timer.stop()
            self._timer = None
            self.engine.draw()


def _opt(args, flag, default):
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def get_plugin(args):
    return DojoReplayPlugin(args)
