# HANDOFF → AG: single-chart entry tagging, confluence zones, label overlay
**Doc:** 047 · **Date:** 2026-07-12 · **Author:** Claude (reviewer/executor, at usage limit) · **Status:** DIRECTIVE for AG
**Plan by:** Moises. Claude audits on return — checklist in §4.

## 1. Moises' plan (the next work, verbatim intent)
1. Tag ALL catalog event entries onto ONE single chart.
2. See which signals CONFOUND (co-fire) in the same areas → confluence zones.
3. Overlay OUR LABELS (the golden auto-labeler dataset) and see how the catalog
   events match up against labeled opportunities.
Context: doc 046 — the reversion/level/divergence family marks +5..+14pt
hour-matched excess 15m amplitude, direction-free. The hypothesis behind the
chart: co-firing clusters may localize the labeled opportunities in time.

## 2. Data (all committed)
- **Entries**: `reports/fps_horizons.parquet` — one row per event: doss, day,
  year, entry_ts (Unix s), is_long, pnl/mfe/mae at 1m/5m/15m/30m/1h/eod.
  ⚠ **ORB-02 rows are MIS-ANCHORED 30min early** (doc 045 bug lives in the
  events file): either add +1800s to ORB entry_ts or drop ORB until the dossier
  re-exports. SEASON-12/RENKO-24 absent (own index spaces).
- **Trade log (management-level, historical)**: `reports/fps_trades.parquet`.
- **Labels**: the ai_auto_labeler v2 golden dataset (30,173 trades / 576 days,
  2026-06-30 journal) — locate under `research/ai_auto_labeler/` outputs; if
  path unclear, grep INDEX.md 2026-06-30 entry.
- Raw bars: `DATA/ATLAS/5s`; canonical stream: `core_v2 FPS` (use_5s_price=True).

## 3. Method guidance + pre-registered pitfalls
- Co-firing: bucket entries into 5-min bins per day; pairwise Jaccard/co-count
  matrix between dossiers; confluence zone = bin with >=k distinct dossiers.
  ⚠ The reversion family's triggers are CORRELATED BY CONSTRUCTION (same price
  geometry) — co-firing alone is NOT independent confirmation; report the
  co-fire matrix so correlated pairs are visible before "confluence" is claimed.
- Label overlay: for each labeled trade entry, distance (in minutes) to nearest
  catalog event, split by dossier family; compare against shuffled-bin baseline
  ONLY of the arithmetic kind (no synthetic responses — house rule, doc 013).
- Day-block CIs for any rate claim; mode-first distributions; raw points.
- Protocol: one numbered comms doc per turn (next = 048), evidence-coupled
  claims (artifact path + pasted output), commit+push each turn, stay on cron
  until released. Root of catalog = protocol/README/folders ONLY.

## 4. Claude's audit checklist on return (do not delete)
1. ORB entry_ts handling (corrected or excluded?).
2. Co-fire matrix present BEFORE confluence claims; correlated-pair caveat.
3. Label-overlay distances vs arithmetic baseline; no synthetic nulls.
4. Any new "finding" — check %>0=1.00-class impossibilities and index-space
   provenance FIRST (six artifact classes, docs 036/040/042/045).
5. entry_ts export for dossiers (standing corrective, doc 045) — done?
6. Journals: every AG turn appended to docs/daily + INDEX.

## 5. Program state one-liner (for cold readers)
Catalog = closed directional null (doc 045) + a real direction-free amplitude-
marking family (doc 046, GARCH-match still pending). Current work = Moises'
confluence/label-overlay chart to test whether catalog events LOCALIZE the
labeled opportunities. FPS is canonical and fast (~130-156k bars/s).
