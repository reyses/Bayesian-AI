# Bayesian-AI Tools Index

## Data Sourcing & Pipeline
- `tools/sourcing/convert_nt8_csv_to_parquet.py`: Converts RAW NT8 CSV dumps into ATLAS_NT8 schema with session-day partitioning and contract roll stitching.
- `DATA/pipeline/build_timeframes.py`: Aggregates 1s and 1m ATLAS parquets into coarser timeframes (5s, 15s, 5m, 1h, etc.) and validates their parity.
- `DATA/pipeline/databento_to_atlas.py`: Downloads and processes raw databento MBP-1 tick/bar data into the ATLAS schema.
