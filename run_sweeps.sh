#!/bin/bash
for f in research/leg_clock/tools/ag_cat_0[2-8]*.py; do
    echo "Running $f"
    .venv_wsl/bin/python "$f"
done
