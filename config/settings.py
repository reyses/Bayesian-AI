"""
Bayesian-AI - System State Control
"""

# Operational Mode: "LEARNING" or "EXECUTE"
OPERATIONAL_MODE = "LEARNING"

# Data Path Configuration
# Raw source dumps moved to D: 2026-07-16 (OneDrive/C: space); ATLAS stays in-repo
RAW_DATA_PATH = "D:/Bayesian-AI-data/DATA/RAW"

# Anchor Date for Training/Simulation (YYYY-MM-DD)
# Determines the start date for data file selection
ANCHOR_DATE = "2025-07-30"

# --- EXECUTION PHYSICS (GLOBAL TRUTH) ---
DEFAULT_BASE_SLIPPAGE = 0.25
DEFAULT_VELOCITY_SLIPPAGE_FACTOR = 0.1
