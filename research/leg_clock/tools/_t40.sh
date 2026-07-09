#!/usr/bin/env bash
PY=.venv_wsl/bin/python
P=research/leg_clock/tools/dev_loop_2025.py
$PY $P --direction fade --trail 40 --tiers 0.995 --hours 9-13 --year 2025 2>&1 | grep dev:
$PY $P --direction fade --trail 40 --tiers 0.995 --hours 9-13 --year 2024 2>&1 | grep dev:
