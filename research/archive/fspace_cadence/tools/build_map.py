"""Composite F-space MAP from enriched Stage-1 R-curves (per JULES_STAGE12_MAP_OVERHAUL.md).

Reads the enriched B2{C,T}map_{REAL,BROWN,FOUR} segment JSONs and builds two decisive curves
per (representation, series), comparing REAL vs Brownian vs Fourier:

  1. EXPLAINABILITY curve: median r2 across all regimes' R-curves at each forward length L
     -> "how cleanly does the F-space explain the path, and how does it distort with horizon."
  2. SURVIVAL curve: fraction of regimes still un-flipped (length >= L) -> the FLIP-TIMING map.

Plus direction summary (P(next=same), up-fraction). Reported MODE-FIRST at fixed L offsets.

NOTE: single Brownian + single Fourier draw here (n=1 null) -> NO p-values yet. Significance
needs the Fourier surrogate ENSEMBLE (spec §4); this is the descriptive distortion map.
"""
import os, json
import numpy as np

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts"))
OFFSETS = [30, 45, 60, 90, 120, 180, 300]   # forward length L (seconds == 1s bars)
RUNS = {
    ('B2C', 'REAL'):  'stage1_B2Cmap_REAL_segments_2024_02_20.json',
    ('B2C', 'BROWN'): 'stage1_B2Cmap_BROWN_segments_2024_02_20_BROWN.json',
    ('B2C', 'FOUR'):  'stage1_B2Cmap_FOUR_segments_2024_02_20_FOUR.json',
    ('B2T', 'REAL'):  'stage1_B2Tmap_REAL_segments_2024_02_20.json',
    ('B2T', 'BROWN'): 'stage1_B2Tmap_BROWN_segments_2024_02_20_BROWN.json',
    ('B2T', 'FOUR'):  'stage1_B2Tmap_FOUR_segments_2024_02_20_FOUR.json',
    ('RUNC', 'REAL'):  'stage1_RUNCmap_REAL_segments_2024_02_20.json',
    ('RUNC', 'BROWN'): 'stage1_RUNCmap_BROWN_segments_2024_02_20_BROWN.json',
    ('RUNC', 'FOUR'):  'stage1_RUNCmap_FOUR_segments_2024_02_20_FOUR.json',
}


def curves(path):
    segs = json.load(open(os.path.join(ART, path)))
    pris = [s for s in segs if s['status'] == 'PRISTINE']
    lengths = np.array([s['length'] for s in pris]) if pris else np.array([0])
    # explainability: r2 samples at each L pooled across all r_curves
    r2_at = {L: [] for L in OFFSETS}
    for s in pris:
        for row in s.get('r_curve', []):
            L = row['L']
            for off in OFFSETS:
                if abs(L - off) <= 2:            # +-2s bucket around the offset
                    r2_at[off].append(row['r2'])
    expl = {L: (np.median(v) if v else float('nan')) for L, v in r2_at.items()}
    # survival: fraction of regimes with length >= L
    n = len(pris)
    surv = {L: float(np.mean(lengths >= L)) for L in OFFSETS}
    # direction
    dirs = np.array([s.get('direction', 0) for s in pris])
    dnz = dirs[dirs != 0]
    same = float(np.mean(dnz[1:] == dnz[:-1])) if len(dnz) > 1 else float('nan')
    upf = float(np.mean(dirs > 0))
    return dict(n=n, med_len=int(np.median(lengths)), expl=expl, surv=surv, same=same, upf=upf)


def main():
    data = {k: curves(v) for k, v in RUNS.items()}
    lines = []
    def p(s): print(s); lines.append(s)

    p("# F-space MAP (descriptive; n=1 null, no p-values yet) — 2024_02_20\n")
    for rep in ('B2C', 'B2T', 'RUNC'):
        r, b, f = data[(rep, 'REAL')], data[(rep, 'BROWN')], data[(rep, 'FOUR')]
        p(f"## {rep}  (REAL n={r['n']} medLen {r['med_len']}s | dir up {r['upf']:.0%} P(next=same) {r['same']:.0%})")
        p("EXPLAINABILITY — median r2 at length L (REAL | BROWN | FOUR | REAL-FOUR gap)")
        for L in OFFSETS:
            g = r['expl'][L] - f['expl'][L]
            p(f"  L={L:4d}s : {r['expl'][L]:+.3f} | {b['expl'][L]:+.3f} | {f['expl'][L]:+.3f} | gap {g:+.3f}")
        p("SURVIVAL — frac of regimes still un-flipped at L (REAL | BROWN | FOUR)")
        for L in OFFSETS:
            p(f"  L={L:4d}s : {r['surv'][L]:.2f} | {b['surv'][L]:.2f} | {f['surv'][L]:.2f}")
        p("")

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports",
                                       "map_b2c_b2t_2024_02_20.md"))
    open(out, 'w').write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
