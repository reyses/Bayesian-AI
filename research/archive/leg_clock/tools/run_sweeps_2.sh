#!/bin/bash
for f in research/leg_clock/tools/ag_cat_09*.py research/leg_clock/tools/ag_cat_10*.py research/leg_clock/tools/ag_cat_11*.py; do
    echo "Running $f"
    $HOME/venvs/bayesian-ai/bin/python "$f"
done
