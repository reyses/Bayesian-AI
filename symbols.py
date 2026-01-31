"""
ProjectX v2.0 - Asset Profiles
Futures contract specifications for risk calculations
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class AssetProfile:
    """Futures contract specifications"""
    ticker: str          # Display name (NQ, ES, MNQ, MES)
    symbol_id: str       # TopstepX/Broker ID (F.US.NQ, F.US.MEP, etc.)
    tick_size: float     # Minimum price movement (0.25)
    tick_value: float    # Dollar value per tick ($5, $1.25, $0.50)
    point_value: float   # Dollar value per full point ($20, $50, $5, $2)
    base_price: float    # Approximate current price (for synthetic data)
    base_vol: float      # Base volatility (for synthetic data generation)

# Full-size contracts
NQ = AssetProfile(
    ticker="NQ",
    symbol_id="F.US.NQ",
    tick_size=0.25,
    tick_value=5.0,      # $5 per tick
    point_value=20.0,    # $20 per point (4 ticks = 1 point)
    base_price=21500.0,
    base_vol=0.35
)

ES = AssetProfile(
    ticker="ES",
    symbol_id="F.US.EP",
    tick_size=0.25,
    tick_value=12.5,     # $12.50 per tick
    point_value=50.0,    # $50 per point
    base_price=5800.0,
    base_vol=0.15
)

# Micro contracts (1/10 size of full contracts)
MES = AssetProfile(
    ticker="MES",
    symbol_id="F.US.MEP",
    tick_size=0.25,
    tick_value=1.25,     # $1.25 per tick (1/10 of ES)
    point_value=5.0,     # $5 per point
    base_price=5800.0,
    base_vol=0.15
)

MNQ = AssetProfile(
    ticker="MNQ",
    symbol_id="F.US.MNQ",
    tick_size=0.25,
    tick_value=0.50,     # $0.50 per tick (1/10 of NQ)
    point_value=2.0,     # $2 per point
    base_price=21500.0,
    base_vol=0.35
)

# Symbol map for easy lookup
SYMBOL_MAP = {
    "NQ": NQ,
    "ES": ES,
    "MES": MES,
    "MNQ": MNQ
}

def calculate_pnl(asset: AssetProfile, entry_price: float, exit_price: float, side: str) -> float:
    """
    Calculate P&L for a trade
    Args:
        asset: AssetProfile
        entry_price: Entry price
        exit_price: Exit price
        side: 'long' or 'short'
    Returns:
        float: P&L in dollars
    """
    price_diff = exit_price - entry_price
    
    if side == 'short':
        price_diff = -price_diff
    
    # Convert price difference to ticks
    ticks = price_diff / asset.tick_size
    
    # Calculate dollar P&L
    pnl = ticks * asset.tick_value
    
    return pnl

def calculate_stop_distance(asset: AssetProfile, stop_ticks: int) -> float:
    """
    Calculate stop loss dollar risk
    Args:
        asset: AssetProfile
        stop_ticks: Number of ticks for stop
    Returns:
        float: Dollar risk
    """
    return stop_ticks * asset.tick_value
