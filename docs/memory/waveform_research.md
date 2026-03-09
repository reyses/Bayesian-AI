# Waveform Screening & Seed Library Research

> Referenced from MEMORY.md. Full journal: `docs/reference/RESEARCH_JOURNAL.txt`

## Waveform Screening (active research)
- **Journal**: `docs/reference/RESEARCH_JOURNAL.txt` — full methodology + insights
- **Key insight**: segment on price/time FIRST, layer 16D physics on top
- Price I-MR: I=close, MR=signed bar-to-bar change, regimes from UCL breaks
- Default mode = price I-MR only; `--full` = adds 16D fractal pipeline
- Signed MR: every change is a potential pattern, sign flips = direction changes
- **Analyses completed**: A-H (screening), I (seed classification, 20 primitives),
  J (adaptive R² sub-types, IQR quality gate, R² ceiling 0.88),
  K (direction prediction: 70.6% accuracy, +16.5% lift, 4h/1h dominant),
  L-P (various), Q (signed magnitude histogram + paired 192D profiles)
- **Key K findings**: 1h_hurst #1, 4h_osc_coh #2, 15m base TF barely top-10
- **Q status**: Sign-first split working. Adaptive sigma + IQR fallback added.
  Need `--analysis-days 120` to get enough samples (default 7d only gives ~460 bars).
  CLI: `--start X` skips to analysis X, `--cache file.npz` saves/loads feature matrix
- **Integration spec**: `docs/JULES_WAVEFORM_SEED_INTEGRATION.md` (5 parts)
  Part 1: seed_library.py (20 shapes), Part 2: 4h worker, Part 3: shape direction P0.5,
  Part 4: GradientBoosting 176D direction model, Part 5: live engine wiring

## Seed Library & Live Worker Architecture (KEY DECISION)
- **The waveform analysis is OFFLINE research** — too slow for live trading
- **Output = a pre-built SEED LIBRARY**: shape templates (mathematical functions
  with fitted parameters) + price models, serialized as a lookup table
- **Workers receive the library at startup**, NOT the raw analysis code
- **Live workflow**: observe N bars -> delta from entry -> match to seed library
  (closest function fit) -> get shape type + predicted direction + magnitude
- **This replaces templates.pkl**: instead of DBSCAN clusters, workers get
  named mathematical shapes (V-reversal, ramp, sigmoid, etc.) with parameters
- **Price model (92% R²) enriches** the shape match for entry precision
- The seed library is the bridge between offline research and live execution

## Shape-First Design Direction
- **Pipeline**: DMI pre-split -> I-MR(DMI diff) -> DBSCAN(vol+ADX), 2D clustering, 16D identity
- **Shape-first**: seed functions (ramp, V, sigmoid) replace DBSCAN clusters
- **Laplacian = shape identifier**: d²p/dt² discriminates shape types
