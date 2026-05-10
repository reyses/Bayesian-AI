# Per-tier retune recommendations

Generated 2026-05-10. Validated on full IS+OOS trade history.

For each NMP-based tier, the optimal |z| band (E1) and the
structurally-negative (direction, 1h_z_se_category) cells (E2 VETO).

---

## FADE_CALM

- baseline: n=21371, total=$684.0, IS=$423.0, OOS=$261.0, PF_WR=0.016
- **E1 band**: |z| in [1.5, 1.8] -> n=5216, total=$1003.0, IS=$478.5, OOS=$524.5, PF_WR=0.096
- **E2 VETO** cells (both IS and OOS negative): [('short', 'neutral')]
- **E1+E2 combined**: n=4702, total=$1090.0, IS=$549.0, OOS=$541.0, PF_WR=0.116
- OOS uplift vs baseline: $280.0

## FADE_MOMENTUM

- baseline: n=12766, total=$2502.0, IS=$2339.5, OOS=$162.5, PF_WR=0.038
- **E1 band**: |z| in [1.5, 1.8] -> n=3293, total=$2638.0, IS=$2347.0, OOS=$291.0, PF_WR=0.153
- E2 VETO: no cells qualify (no structural-loser cells found within band)
- OOS uplift vs baseline: $128.5

## FADE_AGAINST

- baseline: n=2811, total=$1398.5, IS=$1254.0, OOS=$144.5, PF_WR=0.189
- **E1 band**: |z| in [1.5, 2.2] -> n=1116, total=$1447.0, IS=$1362.0, OOS=$85.0, PF_WR=0.531
- E2 VETO: no cells qualify (no structural-loser cells found within band)
- OOS uplift vs baseline: $-59.5

## RIDE_CALM

baseline: total=$-369.0  No optimal z-band found (no band with both IS and OOS positive).

## RIDE_MOMENTUM

baseline: total=$-413.5  No optimal z-band found (no band with both IS and OOS positive).

## RIDE_AGAINST

baseline: total=$-2231.5  No optimal z-band found (no band with both IS and OOS positive).

## NMP_FADE_RAW

- baseline: n=67105, total=$5584.0, IS=$5494.0, OOS=$90.0, PF_WR=0.03
- **E1 band**: |z| in [1.5, 1.8] -> n=16375, total=$3275.5, IS=$3249.0, OOS=$26.5, PF_WR=0.069
- **E2 VETO** cells (both IS and OOS negative): [('short', 'aligned')]
- **E1+E2 combined**: n=13076, total=$4078.5, IS=$3738.5, OOS=$340.0, PF_WR=0.109
- OOS uplift vs baseline: $250.0

## NMP_RIDE_RAW

baseline: total=$-132.5  No optimal z-band found (no band with both IS and OOS positive).
