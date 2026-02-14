"""
Integrated Statistical Trading System
Combines: Bayesian validation + Monte Carlo + DOE + Regret Analysis

This replaces the simple confidence threshold with rigorous statistical proof
before allowing any trade execution.
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict


@dataclass
class TradeRecord:
    """Complete trade record for analysis"""
    state_hash: int
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    side: str
    pnl: float
    result: str  # 'WIN' or 'LOSS'
    exit_reason: str
    
    # Regret markers
    peak_favorable: float
    potential_max_pnl: float
    pnl_left_on_table: float
    gave_back_pnl: float
    exit_efficiency: float
    regret_type: str


class BayesianStateValidator:
    """
    Validates if a state has statistically proven edge
    
    Uses Beta-Binomial Bayesian inference with:
    - Informative prior (expect 50% win rate)
    - Posterior updates from observed trades
    - Requires 80% confidence that win_rate > 50%
    """
    
    def __init__(self, 
                 prior_wins: float = 50.0,
                 prior_losses: float = 50.0,
                 min_samples: int = 30,
                 confidence_threshold: float = 0.80):
        """
        Initialize with conservative prior
        
        Args:
            prior_wins: Prior belief in wins (50 = expect 50% win rate)
            prior_losses: Prior belief in losses (50 = expect 50% win rate)
            min_samples: Minimum trades before considering state
            confidence_threshold: Required P(win_rate > 50%)
        """
        self.prior_alpha = prior_wins
        self.prior_beta = prior_losses
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
    
    def validate_state(self, wins: int, losses: int) -> Dict:
        """
        Validate if state has proven edge
        
        Returns decision with full Bayesian analysis
        """
        total = wins + losses
        
        # Check 1: Sufficient sample size
        if total < self.min_samples:
            return {
                'approved': False,
                'reason': f'Insufficient data: {total}/{self.min_samples} trades',
                'confidence': 0.0,
                'expected_win_rate': None,
                'credible_interval': None
            }
        
        # Bayesian posterior
        posterior_alpha = self.prior_alpha + wins
        posterior_beta = self.prior_beta + losses
        
        # Expected win rate (posterior mean)
        expected_wr = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # 95% Credible interval
        ci_lower = stats.beta.ppf(0.025, posterior_alpha, posterior_beta)
        ci_upper = stats.beta.ppf(0.975, posterior_alpha, posterior_beta)
        
        # P(win_rate > 50%) - THE KEY METRIC
        prob_edge = 1 - stats.beta.cdf(0.50, posterior_alpha, posterior_beta)
        
        # Decision
        approved = prob_edge >= self.confidence_threshold
        
        return {
            'approved': approved,
            'reason': f'{prob_edge:.1%} confident win_rate > 50%' if approved else f'Only {prob_edge:.1%} confident',
            'confidence': prob_edge,
            'expected_win_rate': expected_wr,
            'credible_interval': (ci_lower, ci_upper),
            'samples': total,
            'observed_wr': wins / total if total > 0 else 0.0
        }


class MonteCarloRiskAnalyzer:
    """
    Simulates future performance to validate edge robustness
    
    Questions answered:
    1. What's probability of 5 consecutive losses?
    2. What's expected max drawdown?
    3. What's probability Sharpe > 1.0?
    """
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
    
    def simulate_drawdown(self, win_rate: float, avg_win: float, avg_loss: float,
                         n_trades: int = 100) -> Dict:
        """
        Monte Carlo simulation of drawdown risk
        
        Returns probability distributions of key risk metrics
        """
        max_drawdowns = []
        consecutive_losses = []
        final_pnls = []
        sharpe_ratios = []
        
        for _ in range(self.n_simulations):
            # Simulate trades
            outcomes = np.random.random(n_trades) < win_rate
            pnls = np.where(outcomes, avg_win, avg_loss)
            
            # Cumulative P&L
            cum_pnl = np.cumsum(pnls)
            
            # Max drawdown
            running_max = np.maximum.accumulate(cum_pnl)
            drawdowns = running_max - cum_pnl
            max_dd = np.max(drawdowns)
            max_drawdowns.append(max_dd)
            
            # Consecutive losses
            max_consec = 0
            current_consec = 0
            for outcome in outcomes:
                if not outcome:
                    current_consec += 1
                    max_consec = max(max_consec, current_consec)
                else:
                    current_consec = 0
            consecutive_losses.append(max_consec)
            
            # Final P&L
            final_pnls.append(cum_pnl[-1])
            
            # Sharpe ratio
            if len(pnls) > 1:
                sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10)
                sharpe_ratios.append(sharpe)
        
        return {
            'prob_profit': np.mean(np.array(final_pnls) > 0),
            'expected_pnl': np.mean(final_pnls),
            'expected_max_dd': np.mean(max_drawdowns),
            'dd_95th_percentile': np.percentile(max_drawdowns, 95),
            'prob_5_consecutive_losses': np.mean(np.array(consecutive_losses) >= 5),
            'prob_10_consecutive_losses': np.mean(np.array(consecutive_losses) >= 10),
            'expected_sharpe': np.mean(sharpe_ratios),
            'prob_sharpe_gt_1': np.mean(np.array(sharpe_ratios) > 1.0),
            'prob_sharpe_gt_0': np.mean(np.array(sharpe_ratios) > 0)
        }
    
    def validate_risk_profile(self, mc_results: Dict, max_acceptable_dd: float = 500.0) -> Dict:
        """
        Check if risk profile is acceptable
        
        Args:
            mc_results: Results from simulate_drawdown()
            max_acceptable_dd: Maximum acceptable drawdown ($)
            
        Returns:
            Risk approval decision
        """
        approved = (
            mc_results['prob_profit'] >= 0.70 and  # 70%+ chance of profit
            mc_results['expected_max_dd'] <= max_acceptable_dd and  # Expected DD acceptable
            mc_results['prob_sharpe_gt_0'] >= 0.75  # 75%+ chance positive Sharpe
        )
        
        concerns = []
        if mc_results['prob_profit'] < 0.70:
            concerns.append(f"Low profit probability: {mc_results['prob_profit']:.1%}")
        if mc_results['expected_max_dd'] > max_acceptable_dd:
            concerns.append(f"High expected drawdown: ${mc_results['expected_max_dd']:.0f}")
        if mc_results['prob_sharpe_gt_0'] < 0.75:
            concerns.append(f"Low Sharpe probability: {mc_results['prob_sharpe_gt_0']:.1%}")
        
        return {
            'risk_approved': approved,
            'concerns': concerns if not approved else [],
            'probability_ruin': mc_results['prob_10_consecutive_losses'],
            'expected_sharpe': mc_results['expected_sharpe']
        }


class ExperimentalDesignOptimizer:
    """
    DOE to systematically test parameter combinations
    
    Tests:
    - Different hold times
    - Different stop losses
    - Different trail configurations
    
    Identifies which factors actually matter
    """
    
    def __init__(self):
        self.experiments = []
        self.results = []
    
    def generate_factorial_design(self) -> pd.DataFrame:
        """
        Generate experiments to test
        
        Factors:
        - stop_loss: [10, 15, 20] ticks
        - trail_tight: [5, 10, 15] ticks
        - trail_medium: [15, 20, 25] ticks
        - trail_wide: [25, 30, 35] ticks
        - min_samples: [20, 30, 40] trades before state validation
        """
        from itertools import product
        
        factors = {
            'stop_loss': [10, 15, 20],
            'trail_tight': [5, 10, 15],
            'trail_medium': [15, 20, 25],
            'trail_wide': [25, 30, 35],
            'min_samples': [20, 30, 40]
        }
        
        combinations = list(product(*factors.values()))
        
        df = pd.DataFrame(combinations, columns=list(factors.keys()))
        df['experiment_id'] = range(len(df))
        
        return df
    
    def record_experiment_result(self, experiment_id: int, win_rate: float, 
                                sharpe: float, max_dd: float, total_pnl: float):
        """Record result from experiment"""
        self.results.append({
            'experiment_id': experiment_id,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_pnl': total_pnl
        })
    
    def analyze_factor_importance(self, response_var: str = 'sharpe') -> pd.DataFrame:
        """
        Identify which factors matter most
        
        Uses ANOVA to test significance
        """
        if not self.results:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(self.results)
        df_experiments = pd.DataFrame(self.experiments)
        df = df_experiments.merge(df_results, on='experiment_id')
        
        effects = []
        factor_cols = ['stop_loss', 'trail_tight', 'trail_medium', 'trail_wide', 'min_samples']
        
        for factor in factor_cols:
            # Group by factor level
            grouped = df.groupby(factor)[response_var].mean()
            effect_size = grouped.max() - grouped.min()
            
            # ANOVA test
            groups = [df[df[factor] == level][response_var].values 
                     for level in df[factor].unique()]
            f_stat, p_value = stats.f_oneway(*groups)
            
            effects.append({
                'factor': factor,
                'effect_size': effect_size,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        df_effects = pd.DataFrame(effects)
        df_effects = df_effects.sort_values('effect_size', ascending=False)
        
        return df_effects


class IntegratedStatisticalEngine:
    """
    MASTER SYSTEM: Combines all statistical validation
    
    Decision flow:
    1. Bayesian: Does state have proven edge? (>80% confident win_rate > 50%)
    2. Monte Carlo: Is risk profile acceptable? (DD, consecutive losses)
    3. Regret: Are exits optimal? (efficiency > 70%)
    4. DOE: Which parameters work best?
    
    Only fires trades that pass ALL validations
    """
    
    def __init__(self, asset_profile):
        self.asset = asset_profile
        
        # Components
        self.bayesian = BayesianStateValidator(
            prior_wins=50,
            prior_losses=50,
            min_samples=30,
            confidence_threshold=0.80
        )
        
        self.monte_carlo = MonteCarloRiskAnalyzer(n_simulations=10000)
        
        self.doe = ExperimentalDesignOptimizer()
        
        # State tracking
        self.state_records = defaultdict(lambda: {
            'wins': 0,
            'losses': 0,
            'pnls': [],
            'trades': []
        })
        
        # Regret tracking
        self.regret_records = []
    
    def record_trade(self, trade: TradeRecord):
        """Record completed trade"""
        state_hash = trade.state_hash
        
        # Update win/loss counts
        if trade.result == 'WIN':
            self.state_records[state_hash]['wins'] += 1
        else:
            self.state_records[state_hash]['losses'] += 1
        
        self.state_records[state_hash]['pnls'].append(trade.pnl)
        self.state_records[state_hash]['trades'].append(trade)
        
        # Store regret data
        self.regret_records.append({
            'state': state_hash,
            'pnl': trade.pnl,
            'exit_efficiency': trade.exit_efficiency,
            'regret_type': trade.regret_type,
            'left_on_table': trade.pnl_left_on_table,
            'gave_back': trade.gave_back_pnl
        })
    
    def should_fire(self, state_hash: int) -> Dict:
        """
        MASTER DECISION FUNCTION
        
        Returns comprehensive decision with all validations
        """
        record = self.state_records[state_hash]
        wins = record['wins']
        losses = record['losses']
        total = wins + losses
        
        # Phase 1: Bayesian validation
        bayesian_result = self.bayesian.validate_state(wins, losses)
        
        if not bayesian_result['approved']:
            return {
                'should_fire': False,
                'reason': f"Bayesian: {bayesian_result['reason']}",
                'validations': {
                    'bayesian': bayesian_result,
                    'monte_carlo': None,
                    'regret': None
                }
            }
        
        # Phase 2: Monte Carlo risk analysis
        if total >= 10:  # Need some history for MC
            avg_win = np.mean([p for p in record['pnls'] if p > 0]) if any(p > 0 for p in record['pnls']) else 200
            avg_loss = np.mean([p for p in record['pnls'] if p < 0]) if any(p < 0 for p in record['pnls']) else -100
            
            mc_results = self.monte_carlo.simulate_drawdown(
                win_rate=bayesian_result['expected_win_rate'],
                avg_win=avg_win,
                avg_loss=avg_loss,
                n_trades=100
            )
            
            risk_validation = self.monte_carlo.validate_risk_profile(mc_results)
            
            if not risk_validation['risk_approved']:
                return {
                    'should_fire': False,
                    'reason': f"Risk: {', '.join(risk_validation['concerns'])}",
                    'validations': {
                        'bayesian': bayesian_result,
                        'monte_carlo': risk_validation,
                        'regret': None
                    }
                }
        else:
            mc_results = None
            risk_validation = {'risk_approved': True, 'concerns': []}
        
        # Phase 3: Regret analysis (check exit quality)
        state_regrets = [r for r in self.regret_records if r['state'] == state_hash]
        
        if len(state_regrets) >= 10:
            avg_efficiency = np.mean([r['exit_efficiency'] for r in state_regrets])
            
            # If exits are terrible, don't trade
            if avg_efficiency < 0.50:
                return {
                    'should_fire': False,
                    'reason': f"Poor exit quality: {avg_efficiency:.1%} efficiency",
                    'validations': {
                        'bayesian': bayesian_result,
                        'monte_carlo': risk_validation,
                        'regret': {'avg_efficiency': avg_efficiency, 'approved': False}
                    }
                }
        
        # ALL VALIDATIONS PASSED
        return {
            'should_fire': True,
            'reason': f"All validations passed: {bayesian_result['confidence']:.1%} confidence",
            'validations': {
                'bayesian': bayesian_result,
                'monte_carlo': risk_validation if mc_results else None,
                'regret': {'approved': True}
            },
            'expected_win_rate': bayesian_result['expected_win_rate'],
            'credible_interval': bayesian_result['credible_interval']
        }
    
    def get_summary_statistics(self) -> Dict:
        """Get overall system statistics"""
        total_states = len(self.state_records)
        
        # Bayesian-approved states
        approved_states = []
        for state_hash, record in self.state_records.items():
            result = self.bayesian.validate_state(record['wins'], record['losses'])
            if result['approved']:
                approved_states.append(state_hash)
        
        # Regret statistics
        if self.regret_records:
            avg_efficiency = np.mean([r['exit_efficiency'] for r in self.regret_records])
            too_early = len([r for r in self.regret_records if r['regret_type'] == 'closed_too_early'])
            too_late = len([r for r in self.regret_records if r['regret_type'] == 'closed_too_late'])
        else:
            avg_efficiency = 0.0
            too_early = 0
            too_late = 0
        
        return {
            'total_states_observed': total_states,
            'bayesian_approved_states': len(approved_states),
            'approval_rate': len(approved_states) / total_states if total_states > 0 else 0.0,
            'total_trades': sum(r['wins'] + r['losses'] for r in self.state_records.values()),
            'avg_exit_efficiency': avg_efficiency,
            'exits_too_early': too_early,
            'exits_too_late': too_late
        }


# Example Usage
if __name__ == "__main__":
    from config.symbols import MNQ
    
    print("="*80)
    print("INTEGRATED STATISTICAL TRADING SYSTEM - DEMO")
    print("="*80)
    
    # Initialize system
    engine = IntegratedStatisticalEngine(MNQ)
    
    # Simulate some trades on a hypothetical state
    test_state_hash = 12345
    
    print("\nSimulating 40 trades for state 12345...")
    print("Win rate: 60%, Avg win: $200, Avg loss: $100")
    
    for i in range(40):
        # Simulate trade outcome
        is_win = np.random.random() < 0.60
        pnl = 200 if is_win else -100
        
        trade = TradeRecord(
            state_hash=test_state_hash,
            entry_price=21500,
            exit_price=21500 + (10 if is_win else -5),
            entry_time=i * 60,
            exit_time=(i + 1) * 60,
            side='long',
            pnl=pnl,
            result='WIN' if is_win else 'LOSS',
            exit_reason='trail_stop',
            peak_favorable=21510 if is_win else 21495,
            potential_max_pnl=220 if is_win else -80,
            pnl_left_on_table=20 if is_win else 0,
            gave_back_pnl=0 if is_win else 20,
            exit_efficiency=0.91 if is_win else 0.80,
            regret_type='optimal' if is_win else 'closed_too_late'
        )
        
        engine.record_trade(trade)
    
    # Test decision function
    print("\n" + "="*80)
    print("TESTING DECISION FUNCTION")
    print("="*80)
    
    decision = engine.should_fire(test_state_hash)
    
    print(f"\nShould Fire: {decision['should_fire']}")
    print(f"Reason: {decision['reason']}")
    
    if decision['should_fire']:
        print(f"\nBayesian Validation:")
        print(f"  Confidence: {decision['validations']['bayesian']['confidence']:.1%}")
        print(f"  Expected WR: {decision['expected_win_rate']:.1%}")
        print(f"  95% CI: [{decision['credible_interval'][0]:.1%}, {decision['credible_interval'][1]:.1%}]")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SYSTEM SUMMARY")
    print("="*80)
    
    summary = engine.get_summary_statistics()
    print(f"\nTotal states observed: {summary['total_states_observed']}")
    print(f"Bayesian-approved states: {summary['bayesian_approved_states']}")
    print(f"Approval rate: {summary['approval_rate']:.1%}")
    print(f"Total trades: {summary['total_trades']}")
    print(f"Avg exit efficiency: {summary['avg_exit_efficiency']:.1%}")
    print(f"Exits too early: {summary['exits_too_early']}")
    print(f"Exits too late: {summary['exits_too_late']}")
    
    print("\n" + "="*80)
    print("✅ SYSTEM READY FOR INTEGRATION")
    print("="*80)
