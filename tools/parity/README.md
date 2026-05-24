# tools/_parity — Lookahead / Parity Validation

> Virtual folder. Files in `tools/` root.

| Tool | Purpose |
|---|---|
| [`compare_lookahead_impact.py`](../compare_lookahead_impact.py) | Diff honest vs lookahead-inflated blended trades |
| [`lookahead_impact.py`](../lookahead_impact.py) | Per-day tier distribution vs archived baseline |
| [`parity_check.py`](../parity_check.py) | Live log vs baseline forward-pass, timestamp-by-timestamp |
| [`parity_validate.py`](../parity_validate.py) | FEATURE + TRADE parity in one check |
| [`live_parity_check.py`](../live_parity_check.py) | Build parity features from live bars, compare |
| [`validate_sfe_parity.py`](../validate_sfe_parity.py) | SFE feed_bar() vs batch_compute_states ground truth |
