"""LiveConfig  -- all tunables for the NT8 live trading connector."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveConfig:
    # ── Connection ──────────────────────────────────────────────────────
    nt8_host: str = "127.0.0.1"
    nt8_port: int = 5199
    reconnect_delay_s: float = 3.0
    max_reconnect_attempts: int = 50
    heartbeat_interval_s: float = 5.0

    # ── Instrument ──────────────────────────────────────────────────────
    instrument: str = "MNQ 06-26"     # NT8 instrument name (front month)
    account: str = "Sim101"            # Must match NT8 BayesianBridge account
    asset_ticker: str = "MNQ"         # Maps to config.symbols.SYMBOL_MAP
    point_value: float = 2.0          # $/point  -- MNQ=$2, NQ=$20, ES=$50, MES=$5
    tick_size: float = 0.25           # Min price increment

    # ── Checkpoints (reuses training output) ────────────────────────────
    checkpoint_dir: str = "checkpoints"

    # ── Engine ──────────────────────────────────────────────────────────
    warmup_bars: int = 240            # Bars before first signal (auto-scaled to anchor TF)
    base_resolution_s: int = 5       # Bar size from NT8 (5s chart)
    anchor_tf: str = '5s'            # Primary signal TF — matches training pipeline
    pivot_source: str = 'stream'      # L5: 'stream' = forward pass detector (live);
                                       # 'replay' = inject pivots from production
                                       # parquet (mock/SIM validation).

    # ── Ping-Pong ─────────────────────────────────────────────────────
    ping_pong: bool = False           # Continuous wave-riding with direction refinement
    pp_min_conviction: float = 0.55   # Min belief conviction to flip
    pp_agree_veto: float = 0.60       # If belief still agrees with old dir above this, skip flip
    pp_bias_min_trades: int = 5       # Trades needed before bias override kicks in
    pp_bias_wr_good: float = 0.60     # WR above this = "good" direction
    pp_bias_wr_bad: float = 0.40      # WR below this = "bad" direction (reject flips into it)
    pp_sl_override: int = 0           # Override SL ticks (0 = inherit from exited trade)
    pp_tp_override: int = 0           # Override TP ticks (0 = inherit)
    pp_trail_override: int = 0        # Override trail ticks (0 = inherit)
    pp_max_hold_bars: int = 0         # Override max hold (0 = inherit)

    # ── Risk ────────────────────────────────────────────────────────────
    max_position_size: int = 1        # Contracts per entry (1 per primary/chain)
    max_contracts: int = 3            # Absolute max simultaneous contracts (primary + chains)
    max_daily_loss_usd: float = 0.0    # 0 = disabled (sim mode)
