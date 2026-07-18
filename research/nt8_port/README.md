# NT8 Native Port (Architecture B)

This project contains the execution of the Moises-confirmed Architecture B (Native NinjaScript Port) for the entry system, per spec document `129`.

## Architecture B Details
The mechanical manager needs NO cut logic — entry (combiner P) + R-trigger + B9 sizing is the complete system.

## Project Structure
- `pipeline/`: Core logic (if needed).
- `builders/`: Data/feature builders.
- `tools/`: Analysis, orchestration, and generators (e.g. `generate_golden_vectors.py`).
- `reports/`: Findings and reports.
- `golden/`: Contains the P0 golden parity vectors (`*.parquet`).
