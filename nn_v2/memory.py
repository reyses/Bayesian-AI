"""
Bayesian Memory — accumulates evidence per tree branch.

Keyed on leaf_id (the tree already did the feature binning).
Tracks per branch: wins, losses, PnL, drawdown, hold times.
Persists to disk between iterations.

The memory is the bridge between learning and execution:
  - Tree commits branches -> memory records expectations
  - Trial run produces trades -> memory records outcomes
  - Next iteration -> memory provides updated priors

Usage:
    mem = BayesianMemory()
    mem.commit_branches(branches)         # from tree.py
    mem.record_trade(leaf_id, pnl, ...)   # from trial run
    mem.save('DATA/NMP_TREE/memory.pkl')
    mem = BayesianMemory.load('DATA/NMP_TREE/memory.pkl')
"""
import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
from datetime import datetime


class BranchStats:
    """Accumulated stats for one tree branch."""
    __slots__ = ['leaf_id', 'n_trades', 'wins', 'losses', 'pnl_list',
                 'peak_list', 'held_list', 'dd_running', 'dd_max',
                 'expected_wr', 'expected_ev', 'tradeable', 'iteration',
                 'chain_count', 'exit_reasons', 'entry_match_list',
                 'path_adherence_list']

    def __init__(self, leaf_id: int):
        self.leaf_id = leaf_id
        self.n_trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl_list = []
        self.peak_list = []
        self.held_list = []
        self.dd_running = 0.0  # running cumulative PnL for DD calc
        self.dd_max = 0.0      # worst drawdown seen
        # Priors (from tree training)
        self.expected_wr = 0.0
        self.expected_ev = 0.0
        self.tradeable = True
        self.iteration = 0
        # Chain and path tracking
        self.chain_count = 0              # how many trades had chain updates
        self.exit_reasons = []            # why trades exited
        self.entry_match_list = []        # how well 79D matched entry signature
        self.path_adherence_list = []     # expected vs actual path divergence

    @property
    def wr(self):
        return self.wins / max(self.n_trades, 1)

    @property
    def avg_pnl(self):
        return sum(self.pnl_list) / max(len(self.pnl_list), 1)

    @property
    def total_pnl(self):
        return sum(self.pnl_list)

    @property
    def ev(self):
        if not self.pnl_list:
            return self.expected_ev
        wins = [p for p in self.pnl_list if p > 0]
        losses = [p for p in self.pnl_list if p <= 0]
        if not wins:
            return np.mean(losses) if losses else 0.0
        if not losses:
            return np.mean(wins)
        wr = len(wins) / len(self.pnl_list)
        return wr * np.mean(wins) - (1 - wr) * abs(np.mean(losses))

    def record(self, pnl: float, peak: float = 0.0, held: int = 0,
               chain_length: int = 0, exit_reason: str = '',
               entry_match: float = 0.0, path_adherence: float = 0.0):
        """Record a trade outcome with chain and path data."""
        self.n_trades += 1
        self.pnl_list.append(pnl)
        self.peak_list.append(peak)
        self.held_list.append(held)
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        # Drawdown tracking
        self.dd_running += pnl
        peak_equity = max(0, max(np.cumsum(self.pnl_list)))
        current_dd = self.dd_running - peak_equity
        self.dd_max = min(self.dd_max, current_dd)
        # Chain and path tracking
        if chain_length > 0:
            self.chain_count += 1
        if exit_reason:
            self.exit_reasons.append(exit_reason)
        if entry_match > 0:
            self.entry_match_list.append(entry_match)
        if path_adherence != 0:
            self.path_adherence_list.append(path_adherence)


class BayesianMemory:
    """Accumulates evidence per tree branch across iterations."""

    def __init__(self):
        self.branches: Dict[int, BranchStats] = {}
        self.iteration = 0
        self.history = []  # log of iterations

    def commit_branches(self, branches: list):
        """Commit tree branches as priors. Called after tree.py trains.

        Args:
            branches: list of dicts from tree.py with leaf_id, wr, ev, tradeable, etc.
        """
        self.iteration += 1
        for b in branches:
            lid = b['leaf_id']
            if lid not in self.branches:
                self.branches[lid] = BranchStats(lid)
            bs = self.branches[lid]
            bs.expected_wr = b.get('val_wr', b.get('wr', 0))
            bs.expected_ev = b.get('val_ev', b.get('ev', 0))
            bs.tradeable = b.get('tradeable', False)
            bs.iteration = self.iteration

        n_trade = sum(1 for b in self.branches.values() if b.tradeable)
        n_skip = sum(1 for b in self.branches.values() if not b.tradeable)
        print(f'  Memory: committed {len(branches)} branches '
              f'(iteration {self.iteration}, {n_trade} TRADE, {n_skip} SKIP)')

    def record_trade(self, leaf_id: int, pnl: float, peak: float = 0.0,
                     held: int = 0, chain_length: int = 0,
                     exit_reason: str = '', entry_match: float = 0.0,
                     path_adherence: float = 0.0):
        """Record a trade outcome for a branch."""
        if leaf_id not in self.branches:
            self.branches[leaf_id] = BranchStats(leaf_id)
        self.branches[leaf_id].record(pnl, peak, held, chain_length,
                                       exit_reason, entry_match, path_adherence)

    def should_trade(self, leaf_id: int) -> bool:
        """Should we trade this branch? Based on accumulated evidence."""
        if leaf_id not in self.branches:
            return False
        return self.branches[leaf_id].tradeable

    def get_branch(self, leaf_id: int) -> Optional[BranchStats]:
        """Get stats for a branch."""
        return self.branches.get(leaf_id)

    def evaluate_iteration(self) -> Dict:
        """Evaluate current iteration. Returns stats for refinement decisions."""
        tradeable = [b for b in self.branches.values() if b.tradeable]
        skipped = [b for b in self.branches.values() if not b.tradeable]

        # Branches that should be demoted (tradeable but losing)
        demote = [b for b in tradeable if b.n_trades >= 10 and b.wr < 0.50]
        # Branches that should be promoted (skipped but winning in recent data)
        promote = [b for b in skipped if b.n_trades >= 10 and b.wr > 0.60]

        result = {
            'iteration': self.iteration,
            'n_tradeable': len(tradeable),
            'n_skipped': len(skipped),
            'total_trades': sum(b.n_trades for b in tradeable),
            'total_pnl': sum(b.total_pnl for b in tradeable),
            'avg_wr': np.mean([b.wr for b in tradeable]) if tradeable else 0,
            'demote_candidates': [(b.leaf_id, b.wr, b.total_pnl) for b in demote],
            'promote_candidates': [(b.leaf_id, b.wr, b.total_pnl) for b in promote],
        }
        return result

    def refine(self):
        """Auto-refine: demote losing branches, promote winning ones."""
        eval_result = self.evaluate_iteration()

        changes = []
        for lid, wr, pnl in eval_result['demote_candidates']:
            self.branches[lid].tradeable = False
            changes.append(f'  DEMOTED branch {lid}: WR={wr:.0%}, PnL=${pnl:.0f}')

        for lid, wr, pnl in eval_result['promote_candidates']:
            self.branches[lid].tradeable = True
            changes.append(f'  PROMOTED branch {lid}: WR={wr:.0%}, PnL=${pnl:.0f}')

        if changes:
            print(f'\n  Refinement (iteration {self.iteration}):')
            for c in changes:
                print(c)

        self.history.append(eval_result)
        return eval_result

    def summary(self) -> str:
        """Human-readable summary."""
        tradeable = [b for b in self.branches.values() if b.tradeable]
        skipped = [b for b in self.branches.values() if not b.tradeable]

        lines = [
            f'Bayesian Memory (iteration {self.iteration}):',
            f'  TRADE: {len(tradeable)} branches, '
            f'{sum(b.n_trades for b in tradeable)} trades, '
            f'${sum(b.total_pnl for b in tradeable):.0f}',
            f'  SKIP:  {len(skipped)} branches, '
            f'{sum(b.n_trades for b in skipped)} trades, '
            f'${sum(b.total_pnl for b in skipped):.0f}',
            '',
        ]

        if tradeable:
            lines.append(f'  {"ID":>4} {"N":>5} {"WR":>6} {"EV":>7} {"PnL":>8} {"DD":>7} {"Chain":>5} {"Match":>6}')
            lines.append(f'  {"-"*60}')
            for b in sorted(tradeable, key=lambda x: -x.total_pnl):
                if b.n_trades > 0:
                    avg_match = np.mean(b.entry_match_list) if b.entry_match_list else 0
                    lines.append(f'  {b.leaf_id:>4} {b.n_trades:>5} {b.wr:>5.0%} '
                               f'${b.ev:>6.1f} ${b.total_pnl:>7.0f} ${b.dd_max:>6.0f} '
                               f'{b.chain_count:>5} {avg_match:>5.0%}')

        return '\n'.join(lines)

    def save(self, path: str = 'DATA/NMP_TREE/memory.pkl'):
        """Save to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'  Memory saved: {path}')

    @staticmethod
    def load(path: str = 'DATA/NMP_TREE/memory.pkl') -> 'BayesianMemory':
        """Load from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
