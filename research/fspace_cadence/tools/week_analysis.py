"""Week validation of the flip-timing contrast across 3 models x 5 days x {real, Fourier}.

For each model (B2T tiled, B2C continuous, RunC bar-close-sampled), per day compute survival@45/60s
for real and the Fourier null, the per-day gap (real-null), then a DAY-BLOCK bootstrap 95% CI on the
mean gap across the 5 days.

EXPECTED if "cadence is the lever" holds across the week:
  - B2T, RunC (bar-close): positive gap, reproduces all days, CI excludes 0.
  - B2C (continuous): gap ~0, CI includes 0.
Mode-first; n=5 days -> honestly wide CI.
"""
import os, json
import numpy as np

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts"))
DAYS = ['20', '21', '22', '23', '26']
MODELS = {  # model -> (day1 mapname, days2-5 prefix)
    'B2T (tiled)':       ('B2Tmap',  'WK'),
    'B2C (continuous)':  ('B2Cmap',  'WKBC'),
    'RunC (bar-close)':  ('RUNCmap', 'WKRC'),
}


def paths(mapname, pfx, D):
    if D == '20':
        return (f"stage1_{mapname}_REAL_segments_2024_02_20.json",
                f"stage1_{mapname}_FOUR_segments_2024_02_20_FOUR.json")
    return (f"stage1_{pfx}_2024_02_{D}_REAL_segments_2024_02_{D}.json",
            f"stage1_{pfx}_2024_02_{D}_FOUR_segments_2024_02_{D}_FOUR.json")


def survival(path, k):
    s = json.load(open(os.path.join(ART, path)))
    L = np.array([x['length'] for x in s if x['status'] == 'PRISTINE'])
    return float(np.mean(L >= k)) if len(L) else float('nan')


def boot_ci(gaps, n=4000, seed=20240220):
    rng = np.random.default_rng(seed)
    g = np.array(gaps)
    m = [rng.choice(g, size=len(g), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    out = ["# Week flip-timing contrast — survival@45s, REAL vs FOURIER, 3 models x 5 days (2024_02)\n"]
    def p(s): print(s); out.append(s)
    for model, (mapname, pfx) in MODELS.items():
        gaps = []; per = []
        for D in DAYS:
            rp, fp = paths(mapname, pfx, D)
            try:
                r = survival(rp, 45); f = survival(fp, 45)
            except Exception as e:
                per.append(f"{D}:MISS"); continue
            gaps.append(r - f); per.append(f"{D}:{r:.2f}/{f:.2f}({r-f:+.2f})")
        if len(gaps) >= 2:
            lo, hi = boot_ci(gaps)
            sig = "SIG (excl 0)" if lo > 0 else ("SIG-neg" if hi < 0 else "ns (incl 0)")
            p(f"\n## {model}")
            p("  per-day real/four(gap)@45: " + "  ".join(per))
            p(f"  mean gap {np.mean(gaps):+.3f}  95% day-block CI [{lo:+.3f},{hi:+.3f}]  -> {sig}")
            p(f"  reproduces (all days gap>0)? {'YES' if all(g>0 for g in gaps) else 'NO'}")
        else:
            p(f"\n## {model}: insufficient days ({len(gaps)})")
    op = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports",
                                      "week_3model_contrast_2024_02.md"))
    open(op, 'w').write("\n".join(out) + "\n")
    print(f"\nwrote {op}")


if __name__ == '__main__':
    main()
