"""LiveConfig — all tunables for the NT8 live trading connector."""

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
    instrument: str = "MNQ 03-26"     # NT8 instrument name (front month)
    account: str = "Sim101"           # Sim101 for paper, real account for live
    asset_ticker: str = "MNQ"         # Maps to config.symbols.SYMBOL_MAP

    # ── Checkpoints (reuses training output) ────────────────────────────
    checkpoint_dir: str = "checkpoints"

    # ── Engine ──────────────────────────────────────────────────────────
    warmup_bars: int = 240            # 1h of 15s bars before first signal
    base_resolution_s: int = 15      # Bar size from NT8

    # ── Risk ────────────────────────────────────────────────────────────
    max_position_size: int = 1        # Single contract for paper
    max_daily_loss_usd: float = 200.0
