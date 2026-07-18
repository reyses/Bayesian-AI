#!/usr/bin/env bash
P=research/leg_clock/tools/dev_loop_2025.py
PY=$HOME/venvs/bayesian-ai/bin/python
$PY $P --direction follow --trail 20 --tiers 0.995,0.998 2>&1 | grep 2025dev
$PY $P --direction vel    --trail 20 --tiers 0.995,0.998 2>&1 | grep 2025dev
$PY $P --direction fade   --trail 40 --tiers 0.995,0.998 2>&1 | grep 2025dev
$PY $P --direction fade   --trail 80 --tiers 0.995,0.998 2>&1 | grep 2025dev
