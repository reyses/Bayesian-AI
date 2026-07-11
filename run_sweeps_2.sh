#!/bin/bash
for f in research/leg_clock/tools/ag_cat_09*.py research/leg_clock/tools/ag_cat_10*.py research/leg_clock/tools/ag_cat_11*.py; do
    echo "Running $f"
    .venv_wsl/bin/python "$f"
done
