#!/bin/bash
# Bayesian-AI Training Pipeline Runner
# Automates the workflow: Data Setup -> Validation -> Training -> Inspection

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

DATA_DIR="data/raw"
MODELS_DIR="models"

echo -e "${GREEN}=== Bayesian-AI Training Pipeline ===${NC}"

# 1. Create Data Directory
echo -e "\n${YELLOW}[1/4] Setting up directories...${NC}"
mkdir -p "$DATA_DIR"
mkdir -p "$MODELS_DIR"
echo "Directories ensured: $DATA_DIR, $MODELS_DIR"

# Check for data files
count=$(ls -1 "$DATA_DIR"/*.dbn.zst 2>/dev/null | wc -l)
if [ "$count" -eq 0 ]; then
    echo -e "${RED}WARNING: No .dbn.zst files found in $DATA_DIR.${NC}"
    echo "Please copy your Databento files there before running training."
    echo "Example: cp /path/to/downloads/*.dbn.zst $DATA_DIR/"
    # We continue so tests can run if they don't depend on THIS data (unit tests)
    # But training will likely fail or do nothing.
else
    echo -e "${GREEN}Found $count data files.${NC}"
fi

# 2. Run Tests
echo -e "\n${YELLOW}[2/4] Running Validation Tests...${NC}"

echo "Running test_real_data_velocity.py..."
python -m pytest tests/test_real_data_velocity.py -v || echo -e "${RED}Velocity Gate Test Failed${NC}"

echo "Running test_databento_loading.py..."
python -m pytest tests/test_databento_loading.py -v || echo -e "${RED}Databento Loading Test Failed${NC}"

# 3. Run Training
echo -e "\n${YELLOW}[3/4] Starting Training Orchestrator...${NC}"
if [ "$count" -gt 0 ]; then
    python training/orchestrator.py \
      --data-dir "$DATA_DIR" \
      --iterations 10 \
      --output "$MODELS_DIR"
else
    echo -e "${RED}Skipping training due to missing data.${NC}"
fi

# 4. Inspect Results
echo -e "\n${YELLOW}[4/4] Inspecting Results...${NC}"
MODEL_PATH="$MODELS_DIR/probability_table.pkl"

if [ -f "$MODEL_PATH" ]; then
    python scripts/inspect_results.py "$MODEL_PATH"
else
    echo "No model found at $MODEL_PATH to inspect."
fi

echo -e "\n${GREEN}=== Pipeline Complete ===${NC}"
