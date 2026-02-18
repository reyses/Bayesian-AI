"""
Oracle Engine Configuration
All thresholds and parameters in one place.
"""

# Lookahead windows per timeframe (in bars)
# Chosen to represent ~1-2 "meaningful move" durations at each resolution
ORACLE_LOOKAHEAD_BARS = {
    '1s':  300,   # 5 minutes
    '5s':  120,   # 10 minutes
    '15s':  60,   # 15 minutes
    '1m':   60,   # 1 hour
    '5m':   24,   # 2 hours
    '15m':  16,   # 4 hours
    '1h':    8,   # 8 hours (1 trading day)
    '4h':    6,   # 24 hours
    '1D':    5,   # 5 trading days
    '1W':    4,   # 4 weeks
}

# Classification thresholds
ORACLE_MIN_MOVE_TICKS = 5          # Min move in ticks to be non-noise
ORACLE_HOME_RUN_RATIO = 3.0        # MFE/MAE ratio for Mega classification (+/-2)
ORACLE_SCALP_RATIO = 1.2           # MFE/MAE ratio for Scalp classification (+/-1)

# Marker values (semantic names for clarity)
MARKER_MEGA_LONG = 2
MARKER_SCALP_LONG = 1
MARKER_NOISE = 0
MARKER_SCALP_SHORT = -1
MARKER_MEGA_SHORT = -2

# Template intelligence thresholds
TEMPLATE_MIN_MEMBERS_FOR_STATS = 5      # Need at least N patterns to compute stats
TEMPLATE_TOXIC_RISK_THRESHOLD = 0.70    # risk_score above this = toxic
TEMPLATE_HIGH_WIN_RATE = 0.55           # Above this = promising

# Transition matrix
TRANSITION_MIN_SEQUENCE_GAP_BARS = 1    # Min bars between events to count as transition
TRANSITION_MAX_SEQUENCE_GAP_BARS = 100  # Max bars to look for next event
