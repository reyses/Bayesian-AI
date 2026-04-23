# tools/_physics — Nightmare & Physics Engines

> Virtual folder. Files in `tools/` root.

| Tool | Purpose |
|---|---|
| [`nightmare_ticker.py`](../nightmare_ticker.py) | Zero-lookahead 1s ticker. SFE warmed 300 bars. **Hub** |
| [`nightmare_runner.py`](../nightmare_runner.py) | Per-day wrapper around nightmare_ticker |
| [`nightmare_forward_pass.py`](../nightmare_forward_pass.py) | Nightmare Protocol w/ grounded features |
| [`nightmare_oos_ticker.py`](../nightmare_oos_ticker.py) | Same ticker on OOS data |
| [`nightmare_eda.py`](../nightmare_eda.py) | Deep analysis of losing exit types |
| [`physics_day_chart.py`](../physics_day_chart.py) | Chart one day's physics (price + features) |
| [`physics_exit_chart.py`](../physics_exit_chart.py) | Exit chart — hold until physics signals done |
| [`physics_failure_analysis.py`](../physics_failure_analysis.py) | Where does physics thesis fail? |
| [`physics_funnel_oos.py`](../physics_funnel_oos.py) | IS seed physics → OOS match → enter opposite |
| [`physics_oos_full.py`](../physics_oos_full.py) | Full OOS physics pipeline |
| [`strategy_ticker.py`](../strategy_ticker.py) | 79D + NN + Brain zero-lookahead forward pass |
| [`derive_physics.py`](../derive_physics.py) | Extract entry/exit rules from corrected trades |
