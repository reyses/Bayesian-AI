# tools/_data — Data Pipeline

> Virtual folder — actual `.py` files are in `tools/` root (preserves imports).
> This README lists the tools in this category with one-line purpose.

## Atlas construction
| Tool | Purpose |
|---|---|
| [`atlas_rebuild.py`](../atlas_rebuild.py) | Clean 1s tick spikes + validate all TFs + rebuild ATLAS_FEATURES in one step |
| [`build_feature_atlas.py`](../build_feature_atlas.py) | Pre-compute 13D features per TF → `DATA/ATLAS_FEATURES/` |
| [`build_timeframes.py`](../build_timeframes.py) | 1s → 5/15/30s; 1m → 3/5/15/30m. Validates against 60-bar control |
| [`clean_tick_spikes.py`](../clean_tick_spikes.py) | Interpolate single-bar spikes > THRESHOLD ticks |
| [`setup_oos_atlas.py`](../setup_oos_atlas.py) | IS/OOS split (Jan-Nov 2025 / Dec 2025-Feb 2026) |
| [`rebuild_features.py`](../rebuild_features.py) | Wipe + regenerate FEATURES_5s/ (warm-starts from sibling checkpoint) |

## Data imports
| Tool | Purpose |
|---|---|
| [`databento_to_atlas.py`](../databento_to_atlas.py) | Databento downloads → ATLAS (auto-detects trades/1s/1m/1h/1d) |
| [`rebuild_atlas_databento.py`](../rebuild_atlas_databento.py) | Rebuild ATLAS from Databento `.dbn.zst` (MNQ front-month) |
| [`nt8_export_to_atlas.py`](../nt8_export_to_atlas.py) | NT8 `.txt` exports → daily ATLAS parquet |
| [`nt8_to_atlas.py`](../nt8_to_atlas.py) | NT8 tick export → 1s ATLAS |
| [`nt8_to_parquet.py`](../nt8_to_parquet.py) | NT8 tick data → all TFs (1s–1W) |
| [`convert_nt8_atlas.py`](../convert_nt8_atlas.py) | NT8 history CSV → ATLAS parquet |

## Validation
| Tool | Purpose |
|---|---|
| [`validate_data.py`](../validate_data.py) | Every TF OHLC vs 1s ground truth; fix mismatches |
