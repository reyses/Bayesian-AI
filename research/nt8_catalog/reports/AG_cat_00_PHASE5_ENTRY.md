# Phase-5 Entry Discriminator (leakage-free, PhE only)

Entry-anchor V2 F-space (5s snapshot) -> P(registered response). Thresholds frozen on train year, evaluated once on test year. Day-block bootstrap (4000). Sub-friction gate: |mode| < 2 pts.

> INVERT EV is a MIRROR APPROXIMATION (-magnitude), not a simulated opposite trade.

> Features = 5s-TF entry snapshot, NOT yet the full multi-TF telescoping ladder.

### ATR-09_Statistical_Fade
- train 2024 N=432 (base 0.109) -> test 2025 N=367; 6 features selected
- **ACT**   N=64 (59d) WR=0.11 EV=+2.71pts CI[-7.65,+15.23] mode=-10 not sig
- **INVERT**N=71 (59d) WR=0.90 EV=+2.36pts CI[-7.25,+10.00] mode=+11 not sig

### FIB-17_Confluence
- train 2024 N=32 (base 0.031) -> test 2025 N=42; 5 features selected
- **ACT**   N=3 (3d) WR=0.00 EV=-31.75pts CI[-46.00,-10.25] mode=-46 [UNDERPOWERED] not sig
- **INVERT**N=35 (35d) WR=0.89 EV=-3.89pts CI[-22.72,+10.57] mode=+11 not sig

### VA-13_Rotation
- train 2024 N=82 (base 0.110) -> test 2025 N=50; 1 features selected
- **ACT**   N=5 (5d) WR=0.40 EV=+7.90pts CI[-7.85,+31.50] mode=-14 [UNDERPOWERED] not sig
- **INVERT**N=3 (3d) WR=1.00 EV=+44.25pts CI[+2.00,+125.00] mode=+2 [UNDERPOWERED] not sig
