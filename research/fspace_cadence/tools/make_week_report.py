"""Generate a visual report (PNGs + GIF + markdown) for the week cadence/flip-timing validation."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "assets"))
os.makedirs(OUT, exist_ok=True)
DAYS = ['20', '21', '22', '23', '26']
MODELS = {'B2T (tiled)': ('B2Tmap', 'WK', '#1f77b4'),
          'RunC (bar-close)': ('RUNCmap', 'WKRC', '#2ca02c'),
          'B2C (continuous)': ('B2Cmap', 'WKBC', '#d62728')}
Ls = np.arange(30, 160)

def paths(mapname, pfx, D):
    if D == '20':
        return (f"stage1_{mapname}_REAL_segments_2024_02_20.json",
                f"stage1_{mapname}_FOUR_segments_2024_02_20_FOUR.json")
    return (f"stage1_{pfx}_2024_02_{D}_REAL_segments_2024_02_{D}.json",
            f"stage1_{pfx}_2024_02_{D}_FOUR_segments_2024_02_{D}_FOUR.json")

def lengths(path):
    s = json.load(open(os.path.join(ART, path)))
    return np.array([x['length'] for x in s if x['status'] == 'PRISTINE'])

def surv_curve(L, grid):
    return np.array([np.mean(L >= g) if len(L) else np.nan for g in grid])

def boot_ci(g, n=4000, seed=20240220):
    rng = np.random.default_rng(seed); g = np.array(g)
    m = [rng.choice(g, len(g), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

# ---- load everything ----
data = {}  # (model, D) -> (realL, fourL)
for model, (mn, pfx, c) in MODELS.items():
    for D in DAYS:
        rp, fp = paths(mn, pfx, D)
        data[(model, D)] = (lengths(rp), lengths(fp))

# ===== FIG 1: pooled survival curves (3 panels) =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, (model, (mn, pfx, c)) in zip(axes, MODELS.items()):
    realL = np.concatenate([data[(model, D)][0] for D in DAYS])
    fourL = np.concatenate([data[(model, D)][1] for D in DAYS])
    ax.plot(Ls, surv_curve(realL, Ls), color=c, lw=2.4, label='REAL')
    ax.plot(Ls, surv_curve(fourL, Ls), color='gray', lw=2.0, ls='--', label='Fourier null')
    ax.axvline(45, color='k', ls=':', alpha=0.4)
    ax.set_title(model); ax.set_xlabel('regime length L (s)'); ax.grid(alpha=0.25); ax.legend()
axes[0].set_ylabel('fraction of regimes still un-flipped')
fig.suptitle('Flip-timing survival — REAL vs Fourier null (week pooled, 2024_02_20..26)', fontweight='bold')
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_survival_pooled.png", dpi=110); plt.close(fig)

# ===== FIG 2: per-day @45s gap lines =====
fig, ax = plt.subplots(figsize=(8, 4.6))
gaps = {}
for model, (mn, pfx, c) in MODELS.items():
    g = []
    for D in DAYS:
        rL, fL = data[(model, D)]
        g.append(np.mean(rL >= 45) - np.mean(fL >= 45))
    gaps[model] = g
    ax.plot(DAYS, g, '-o', color=c, lw=2.2, label=model)
ax.axhline(0, color='k', lw=1)
ax.set_xlabel('day (2024_02_)'); ax.set_ylabel('survival@45s gap (REAL - Fourier)')
ax.set_title('Does the edge reproduce across the week?', fontweight='bold'); ax.grid(alpha=0.25); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_perday_gap.png", dpi=110); plt.close(fig)

# ===== FIG 3: forest plot (mean gap +/- day-block CI) =====
fig, ax = plt.subplots(figsize=(8, 3.2))
ys = []
for i, (model, g) in enumerate(gaps.items()):
    lo, hi = boot_ci(g); m = np.mean(g)
    sig = lo > 0
    col = '#2ca02c' if sig else '#888888'
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt='o', color=col, capsize=5, lw=2, ms=8)
    ax.text(hi + 0.01, i, f"{m:+.3f} [{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'ns'}", va='center', fontsize=9)
    ys.append(model)
ax.axvline(0, color='k', lw=1.2)
ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys); ax.set_xlabel('mean survival@45s gap (REAL - Fourier), 95% day-block CI')
ax.set_title('Significance of the flip-timing edge', fontweight='bold'); ax.grid(alpha=0.25, axis='x')
ax.set_xlim(-0.1, 0.45); fig.tight_layout(); fig.savefig(f"{OUT}/fig3_forest.png", dpi=110); plt.close(fig)

# ===== GIF: survival curves day-by-day (reproduction) =====
fig, ax = plt.subplots(figsize=(7.5, 4.6))
def frame(i):
    ax.clear()
    D = DAYS[i]
    for model, (mn, pfx, c) in MODELS.items():
        rL, fL = data[(model, D)]
        ax.plot(Ls, surv_curve(rL, Ls), color=c, lw=2.4, label=f'{model} REAL')
        ax.plot(Ls, surv_curve(fL, Ls), color=c, lw=1.3, ls='--', alpha=0.6)
    ax.axvline(45, color='k', ls=':', alpha=0.4)
    ax.set_ylim(0, 1); ax.set_xlim(Ls[0], Ls[-1]); ax.grid(alpha=0.25)
    ax.set_xlabel('regime length L (s)'); ax.set_ylabel('fraction un-flipped')
    ax.set_title(f'Day 2024_02_{D}  —  solid=REAL, dashed=Fourier null', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
anim = FuncAnimation(fig, frame, frames=len(DAYS), interval=900)
anim.save(f"{OUT}/week_survival.gif", writer=PillowWriter(fps=1.2)); plt.close(fig)

print("figures + gif written to", OUT)
print("files:", os.listdir(OUT))
