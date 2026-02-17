"""
Bayesian-AI - System State Control
"""

# Operational Mode: "LEARNING" or "EXECUTE"
OPERATIONAL_MODE = "LEARNING"

# Data Path Configuration
RAW_DATA_PATH = "DATA/RAW"

# Anchor Date for Training/Simulation (YYYY-MM-DD)
# Determines the start date for data file selection
ANCHOR_DATE = "2025-07-30"

# --- EXECUTION PHYSICS ---
DEFAULT_BASE_SLIPPAGE = 0.25
DEFAULT_VELOCITY_SLIPPAGE_FACTOR = 0.1

# --- TRAINING OPTIMIZATION ---
FISSION_SUBSET_SIZE = 50
INDIVIDUAL_OPTIMIZATION_ITERATIONS = 20
