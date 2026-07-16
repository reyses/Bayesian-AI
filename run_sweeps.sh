#!/bin/bash
for f in research/leg_clock/tools/ag_cat_0[2-8]*.py; do
    echo "Running $f"
    $HOME/venvs/bayesian-ai/bin/python "$f"
done
