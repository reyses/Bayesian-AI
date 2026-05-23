"""Session statistics, trade log, drawdown tracking, and session report."""

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    pnl: float = 0.0
    wins: int = 0
    trades: int = 0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    consec_losses: int = 0
    consec_loss_dollars: float = 0.0
    max_consec_losses: int = 0
    max_consec_loss_dollars: float = 0.0
    consec_wins: int = 0
    peak_equity: float = 0.0
    session_drawdown: float = 0.0
    max_session_drawdown: float = 0.0
    exit_buckets: dict = field(default_factory=lambda: {
        'reversed': 0, 'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0, 'q100plus': 0})


class SessionTracker:
    """Tracks session PnL, drawdowns, trade log, and writes reports."""

    def __init__(self, config):
        self.stats = SessionStats()
        self.trade_log: list[dict] = []
        self._cfg = config
        self._start_time = time.time()

    def record_skip(self, price: float, direction: str, reason: str, prob: float = 0.0):
        """Record a skipped trade (CNN veto, filter reject, etc.)."""
        if not hasattr(self, 'skip_log'):
            self.skip_log = []
        self.skip_log.append({
            'time': time.strftime('%H:%M:%S'),
            'price': price,
            'direction': direction,
            'reason': reason,
            'prob': prob,
        })

    def record_trade(self, pnl: float, trade_info: dict):
        """Record a completed trade. Updates all stats atomically."""
        s = self.stats
        s.pnl += pnl
        s.trades += 1

        if pnl > 0:
            s.wins += 1
            s.gross_win += pnl
            s.consec_wins += 1
            s.consec_losses = 0
            s.consec_loss_dollars = 0.0
        else:
            s.gross_loss += pnl
            s.consec_losses += 1
            s.consec_loss_dollars += abs(pnl)
            s.consec_wins = 0
            s.max_consec_losses = max(s.max_consec_losses, s.consec_losses)
            s.max_consec_loss_dollars = max(s.max_consec_loss_dollars,
                                            s.consec_loss_dollars)

        if s.pnl > s.peak_equity:
            s.peak_equity = s.pnl
        s.session_drawdown = s.peak_equity - s.pnl
        s.max_session_drawdown = max(s.max_session_drawdown, s.session_drawdown)

        # Capture bucket (MFE-based exit quality)
        mfe_ticks = trade_info.get('mfe_ticks', 0)
        pnl_ticks = trade_info.get('pnl_ticks', 0)
        if mfe_ticks > 0:
            capture = pnl_ticks / mfe_ticks * 100
        else:
            capture = 0.0 if pnl <= 0 else 100.0
        if capture > 100:     s.exit_buckets['q100plus'] += 1
        elif capture >= 75:   s.exit_buckets['q4'] += 1
        elif capture >= 50:   s.exit_buckets['q3'] += 1
        elif capture >= 25:   s.exit_buckets['q2'] += 1
        elif capture > 0:     s.exit_buckets['q1'] += 1
        else:                 s.exit_buckets['reversed'] += 1

        trade_info['capture'] = capture
        self.trade_log.append(trade_info)

    @property
    def win_rate(self) -> float:
        return (self.stats.wins / self.stats.trades
                if self.stats.trades > 0 else 0.0)

    @property
    def profit_factor(self) -> float:
        return (self.stats.gross_win / abs(self.stats.gross_loss)
                if self.stats.gross_loss != 0 else 0.0)

    def write_report(self, gate_stats: dict, brain_dir_bias: dict,
                     account_snapshot: dict, pp_flip_count: int = 0,
                     anchor_tf: str = '15s', anchor_depth: int = 8,
                     bar_count: int = 0) -> str:
        """Write session summary to reports/live/. Returns path."""
        from core_v2.report_engine import format_scorecard
        s = self.stats
        report_dir = os.path.join('reports', 'live')
        os.makedirs(report_dir, exist_ok=True)
        _date = time.strftime('%Y%m%d')
        path = os.path.join(report_dir, f'session_{_date}.txt')

        dur = time.time() - self._start_time
        dur_h = int(dur // 3600)
        dur_m = int((dur % 3600) // 60)
        pf = s.gross_win / abs(s.gross_loss) if s.gross_loss != 0 else 0.0
        avg = s.pnl / s.trades if s.trades > 0 else 0.0

        # Shared scorecard (exit/direction/hold breakdowns via report_engine)
        # Normalize trade_log field names for report_engine compatibility
        _norm_trades = []
        for t in self.trade_log:
            _norm_trades.append({
                **t,
                'exit_reason': t.get('reason', t.get('exit_reason', '?')),
                'bars_held': t.get('bars', t.get('bars_held', 0)),
            })
        scorecard = format_scorecard(
            _norm_trades,
            mode='LIVE',
            start_date=time.strftime('%Y-%m-%d'),
            run_ts=time.strftime('%Y-%m-%d %H:%M:%S'),
        )

        L = []
        # Live-specific header
        L.append("=" * 72)
        L.append(f"LIVE SESSION REPORT  (run: {time.strftime('%Y-%m-%d %H:%M:%S')})")
        L.append(f"  Account:    {self._cfg.account}")
        L.append(f"  Instrument: {self._cfg.instrument}")
        L.append(f"  Anchor TF:  {anchor_tf}  (depth={anchor_depth})")
        L.append(f"  Duration:   {dur_h}h {dur_m}m  ({bar_count} bars)")
        L.append("=" * 72)
        L.append("")

        # Drawdown (live-specific: consec loss tracking)
        L.append("── DRAWDOWN & SURVIVAL ──")
        L.append(f"  Max Consecutive Losses: {s.max_consec_losses}")
        L.append(f"  Max Consec Loss $:      ${s.max_consec_loss_dollars:,.2f}")
        L.append(f"  Peak Equity:            ${s.peak_equity:+,.2f}")
        L.append(f"  Max Session Drawdown:   ${s.max_session_drawdown:,.2f}")
        L.append("")

        # Shared scorecard body
        L.append(scorecard)

        # ── Ping-pong direction refinement ──
        if pp_flip_count > 0 or brain_dir_bias:
            L.append("")
            L.append("=" * 72)
            L.append("PING-PONG DIRECTION REFINEMENT")
            L.append("=" * 72)
            L.append(f"  Flip count: {pp_flip_count}")
            if brain_dir_bias:
                L.append(f"  {'Template':<20} {'LONG WR':>10} {'LONG N':>8} "
                         f"{'SHORT WR':>10} {'SHORT N':>8}")
                L.append("  " + "-" * 58)
                for tid, b in sorted(brain_dir_bias.items(),
                                     key=lambda x: sum(x[1].values()),
                                     reverse=True):
                    lw, ll = b['long_w'], b['long_l']
                    sw, sl = b['short_w'], b['short_l']
                    lt, st = lw + ll, sw + sl
                    l_wr = f"{lw/lt:.0%}" if lt > 0 else "n/a"
                    s_wr = f"{sw/st:.0%}" if st > 0 else "n/a"
                    L.append(f"  {str(tid):<20} {l_wr:>10} {lt:>8} "
                             f"{s_wr:>10} {st:>8}")

        # ── Gate rejection funnel ──
        L.append("")
        L.append("=" * 72)
        L.append("GATE REJECTION FUNNEL")
        L.append("=" * 72)
        gs = gate_stats
        _total = gs.get('candidates', gs.get('gate0_skip', 0)
                        + gs.get('traded', 0)) or 1
        L.append(f"  Bars with candidates:           {gs.get('bars_seen', 0):>8,}")
        L.append(f"  Total candidates evaluated:     {_total:>8,}")
        _pct = lambda n: f"{n/_total*100:.1f}%"
        L.append(f"    Pattern Quality  (headroom/physics):   "
                 f"{gs.get('gate0_skip', 0):>8,}  ({_pct(gs.get('gate0_skip', 0))})")
        L.append(f"    Depth Filter     (depth blacklist):    "
                 f"{gs.get('gate0_5_skip', 0):>8,}  ({_pct(gs.get('gate0_5_skip', 0))})")
        L.append(f"    Template Match   (cluster dist):       "
                 f"{gs.get('gate1_nomatch', gs.get('gate1_skip', 0)):>8,}  "
                 f"({_pct(gs.get('gate1_nomatch', gs.get('gate1_skip', 0)))})")
        L.append(f"    Brain Reject     (unprofitable):       "
                 f"{gs.get('gate2_brain', gs.get('gate2_skip', 0)):>8,}  "
                 f"({_pct(gs.get('gate2_brain', gs.get('gate2_skip', 0)))})")
        L.append(f"    Low Conviction   (belief too weak):    "
                 f"{gs.get('gate3_conviction', gs.get('gate3_skip', 0)):>8,}  "
                 f"({_pct(gs.get('gate3_conviction', gs.get('gate3_skip', 0)))})")
        L.append(f"    Screening Filter (fission/hours):      "
                 f"{gs.get('gate3_5_screening', gs.get('gate3_5_skip', 0)):>8,}  "
                 f"({_pct(gs.get('gate3_5_screening', gs.get('gate3_5_skip', 0)))})")
        L.append(f"    Direction Unclear (dir conf too low):   "
                 f"{gs.get('gate4_direction', gs.get('gate4_skip', 0)):>8,}  "
                 f"({_pct(gs.get('gate4_direction', gs.get('gate4_skip', 0)))})")
        L.append(f"    Passed all gates -> traded:            "
                 f"{gs.get('traded', 0):>8,}  ({_pct(gs.get('traded', 0))})")

        # ── Session equity ──
        acct = account_snapshot
        L.append("")
        L.append("=" * 72)
        L.append("ACCOUNT SNAPSHOT")
        L.append("=" * 72)
        L.append(f"  Cash Value:      ${acct.get('cash', 0):>12,.2f}")
        L.append(f"  Unrealized PnL:  ${acct.get('unrealized', 0):>+12,.2f}")
        L.append(f"  Net Liquidation: ${acct.get('net_liq', 0):>12,.2f}")
        L.append(f"  Profit Factor:   {pf:.2f}")
        L.append(f"  Avg PnL/trade:   ${avg:+,.2f}")
        L.append(f"  Gross Win:       ${s.gross_win:+,.2f}")
        L.append(f"  Gross Loss:      ${abs(s.gross_loss):,.2f}")

        # ── Trade log ──
        L.append("")
        L.append("=" * 72)
        L.append("TRADE LOG")
        L.append("=" * 72)
        if self.trade_log:
            L.append(f"  {'#':>3}  {'Time':<10} {'Side':<6} {'Entry':>10} "
                     f"{'Exit':>10} {'PnL':>10} {'Reason':<14} {'Bars':>5}")
            L.append("  " + "-" * 68)
            cum = 0.0
            for i, t in enumerate(self.trade_log, 1):
                cum += t['pnl']
                L.append(
                    f"  {i:>3}  {t['time']:<10} {t['side']:<6} "
                    f"{t['entry']:>10,.2f} {t['exit']:>10,.2f} "
                    f"${t['pnl']:>+9,.2f} {t['reason']:<14} {t['bars']:>5}")
            L.append("  " + "-" * 68)
            L.append(f"  {'':>3}  {'':10} {'':6} {'':10} {'TOTAL':>10} "
                     f"${s.pnl:>+9,.2f}")
        else:
            L.append("  No trades this session.")

        L.append("")
        L.append("=" * 72)

        # Skip log (CNN vetoes, filter rejects)
        if hasattr(self, 'skip_log') and self.skip_log:
            L.append("")
            L.append("=" * 72)
            L.append("SKIP LOG (vetoed entries)")
            L.append("=" * 72)
            L.append(f"    #  Time       Dir      Price       Reason                  Prob")
            L.append("  " + "-" * 68)
            for _si, sk in enumerate(self.skip_log, 1):
                L.append(f"  {_si:>3}  {sk['time']:<10} {sk['direction']:<8} "
                         f"{sk['price']:>10,.2f}  {sk['reason']:<24} {sk['prob']:.2f}")
            L.append(f"  Total skips: {len(self.skip_log)}")

        with open(path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(L) + '\n')
        logger.info(f"Session report saved: {path}")
        return path
