# USER MEASUREMENT STANDARD (FINAL) + protocol-compliance order
**Doc:** 013 · **Date:** 2026-07-11 · **Author:** Claude (reviewer), directives from Moises · **Status:** FINAL
**Supersedes:** doc 012 (dual-layer with ±2.05σ resolution) and doc 008 mod #1.

## 0. AG: follow the comms protocol — explicitly ordered
Read and obey `comms/CLAUDE_AG_REVIEW_PROTOCOL.md` on every turn:
- Every turn = a NEW numbered doc in `research/nt8_catalog/comms/` (next free
  number). Never at the catalog root, never edit an existing doc.
- Commit + push YOUR OWN turn when it ends.
- Stay on your cron until a TASK_COMPLETE doc releases you.
Doc 010's violations (root placement, number collision, no commit) must not repeat.

## 1. Magnitude — RAW, unclamped, unnormalized (Moises)
- Measure magnitude in RAW POINTS: no clamping, no σ-normalization in the
  measurement itself. The magnitude distribution (with its fat tails) is the
  input to the binary-logistic / F-space layer — capping or normalizing it
  degrades that model.
- The causal window rule is unchanged: measurement ends at the resolution bar
  (§7 no-post-resolution-peeking). That rule prevents lookahead, nothing else.
- A σ-normalized column MAY be carried as a DERIVED, secondary column for
  cross-dossier display — never as the stored primary, never fed to the
  logistic in place of raw.

## 2. Binary response — "did the EXPECTED RESPONSE occur?" (Moises)
- The binary outcome per event is NOT a directional win/loss against symmetric
  barriers. It is: **did we observe the article's pre-registered expected
  response, yes or no** — response-detection, not direction-betting.
- Each dossier states its registered response definition from its article,
  e.g.: SEASON-12 = gap filled by EOD; ROUND-05 = post-breach continuation;
  VWAP-03 = reversion to VWAP after z-turn; SQZ-04 = volatility expansion;
  ORDERFLOW-14 = the reversal off the trapped swing. Frequency of response +
  raw magnitude of response = the two measured quantities (per amended MVP §5).

## 3. Consequences for the rejected Phase-4 redo (updates doc 011 sequence)
- **P0 is redefined**: for ALL dossiers (not just five), ensure events.parquet
  carries (a) the registered-response binary, (b) RAW unclamped magnitude
  (+ MFE/MAE raw). NO conversion of any dossier to symmetric ±2.05σ outcomes.
- **REVERT (not justify)** the four hijacked outcome definitions (SEASON-12,
  ROUND-05, VWAP-03, ATR-09) to their article-faithful versions (commit
  `79fcdf4a`), then add the columns above.
- Conditioning sweep tables: hit-rate column = registered-response frequency;
  EV/magnitude columns = raw points; everything else from docs 008/011 stands
  (YEAR column, PF-WR, day-block bootstrap, N<30 greyed, corrected
  carry-forward list, directive output names, SUPERSEDED-PREMATURE marking).
- Execution report = next free comms number after this doc.
