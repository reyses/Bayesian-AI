# FABLE-5 Review — VP-01 proof of concept

**Verdict: architecture APPROVED as the template (test folder + OQ trace +
two-part EV + bootstrap + figures). Five corrections required before the
result is meaningful or the pattern scales.**

## Bugs (fix before rerun)

### B1 — Wrong "open": Globex, not the session open the article means
`open_price = prices[0]` = the first 5s bar of the ATLAS file = **17:00 CT
the PRIOR evening** (files run 23:00 UTC → 22:00 UTC). Value-area setups in
the source article are framed off the DAY-SESSION open (8:30 CT). Right now
every setup is classified by where the *Globex* open sits vs yesterday's
profile — a different (and much earlier) event. Per the directive's own
rule: re-read `3-volume-profile-trading-strategies.md` and register the
open definition the article actually uses; then locate the 8:30 CT bar for
the open.

### B2 — Magnitude uses post-resolution data (lookahead inside the metric)
On a target hit, `magnitude = (max(path) - p0)/σ` where `path` is the FULL
60-min horizon — including everything AFTER the trade resolved (same for
losses with `min(path)`). Consequences: magnitudes are horizon-MFE, not
realized response; EV = P(win)·max-excursion − P(loss)·max-adverse is not
achievable by any exit and inflates the number. Fix: magnitude measured only
UP TO the resolution bar (first touch), or defined explicitly as "response
excursion until resolution".

## Methodology gaps (directive requirements)

### G1 — No null anchor
First-touch ±kσ symmetric has a clean random-walk base of 50% — state it,
and add the matched null (same-day/hour non-event bars) plus the
phantom-level control (jittered POC/VAH/VAL): the "any nearby line bounces"
trap is measured and real in this repo. 48.8% and 61.1% win rates mean
nothing without these; 61% on N=18 is ±23pp.

### G2 — 2024 only
Both-years rule. 2025 must run before any verdict line exists.

### G3 — σ definition off-standard
σ = std of 5s close-to-close diffs is a nonstandard unit (hence the odd
"19.7σ" magnitudes). House standard: trailing 1m regression residual σ (see
`research/level_hold/tools/level_hold_study.py::rolling_ols_bands`). Use it
so magnitudes are comparable across all concept tests.

## Report polish
- Add: registered response (from the article, cited), N-per-day context,
  MODE of magnitude alongside median, bimodality flag, explicit verdict line,
  and "NOT significant" wording where the CI includes 0 (all three EVs here
  include 0 — the current table reads as positive findings).
- Image link is an absolute `file:///C:/...` path — make it relative
  (`assets/...`) so reports render anywhere; keep the figure inside the test
  folder or reports/assets consistently (currently duplicated in both).

## What's right (keep in the template)
Self-contained `tests/<ID>_<name>/` folder; the OQ trace script (manual
verification of computed levels on named days); event rows with day/setup/
mode tags; median-based EV with bootstrap CI; per-setup histograms.
