#!/bin/bash
cd /home/reyse/Bayesian-AI

# Update code
git pull

# Activate environment
if [ -d "$HOME/venv" ]; then
    source $HOME/venv/bin/activate
fi

# Run the year pipeline
echo "Running the year pipeline..."
python3 -m pip install tqdm psutil group-lasso
python3 "research/Regression segments/run_year.py"

echo "Zipping the final output to save space..."
zip artifacts/stage2_year_segments.zip artifacts/stage2_year_segments.json

echo "All done! VM will remain online."
