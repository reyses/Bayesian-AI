"""Gemma vision-triage of trade-path plots via a local OpenAI-compatible endpoint
(Ollama / OpenClaw's backend). Few-shot TAUGHT from the 3-way-verified edge-case manifest,
then classifies the full set. Gemma FLAGS; humans + Claude + Gemini already set ground truth.

Setup (user, local):
  curl -fsSL https://ollama.com/install.sh | sh   # or Windows installer
  ollama serve
  ollama pull gemma3:12b        # vision-capable, ~8GB Q4 — FITS a 12GB RTX 3060 (NOT 27B)

OpenClaw config (for interactive agentic drilling — same endpoint):
  llm: {name: local-ollama, type: openai-compatible, base_url: http://localhost:11434/v1,
        model: gemma3:12b, timeout_ms: 60000}

Run (after Ollama is serving): python research/gemma_triage.py
  --label-col verify_consensus   (the 3-way-verified column; falls back to verify_claude)
"""
import os, sys, glob, json, base64, argparse
import urllib.request
import pandas as pd

BASE_URL = 'http://localhost:11434/v1'      # OpenClaw/Ollama OpenAI-compatible endpoint
MODEL = 'gemma3:12b'
EDGE_DIR = 'reports/findings/edge_cases_clean'
FEWSHOT_PER = 1                              # verified exemplars per archetype in the prompt

ARCHES = ['CLEAN_RIDE', 'GAVE_BACK', 'CHOP', 'STOPPED', 'SMALL_WIN', 'SMALL_LOSS']
SYSTEM = (
 "You inspect trade-trajectory plots from a futures backtest. Each plot shows signed PnL (points) "
 "over the life of one trade: green dot=entry(0), red dot=exit, orange ^=peak favorable (MFE), "
 "purple v=worst (MAE). Classify the SHAPE into exactly one archetype and judge entry/exit quality.\n"
 "Archetypes:\n"
 "- CLEAN_RIDE: smooth directional move kept most of the way to exit (a real trend ride).\n"
 "- GAVE_BACK: rose to a large peak then bled most of it back before exit (exit too wide).\n"
 "- CHOP: flat/low-amplitude noise, never developed a move (false-start entry).\n"
 "- STOPPED: went against quickly and hit a stop (mistimed entry).\n"
 "- SMALL_WIN / SMALL_LOSS: marginal outcome, no clear archetype.\n"
 "Return STRICT JSON only: "
 '{"archetype":"<one>","entry_ok":true/false,"exit_ok":true/false,"flag":"<short note or empty>"}')


def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def img_content(path):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(path)}"}}


def build_fewshot(man, label_col):
    """One verified exemplar image+label per archetype (the teaching set)."""
    msgs = []
    for a in ARCHES:
        sub = man[(man['arch'] == a) & man[label_col].astype(str).str.len().gt(0)]
        if not len(sub):
            continue
        r = sub.iloc[0]
        p = os.path.join(EDGE_DIR, r['plot'])
        if not os.path.exists(p):
            continue
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"Classify ({a} exemplar):"},
                                                  img_content(p)]})
        msgs.append({"role": "assistant", "content": json.dumps(
            {"archetype": a, "entry_ok": 'ok' in str(r.get('proposed_entry', '')).lower(),
             "exit_ok": 'ok' in str(r.get('proposed_exit', '')).lower(),
             "flag": str(r[label_col])[:60]})})
    return msgs


def call(messages, timeout=120):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0,
                       "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer ollama"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out['choices'][0]['message']['content']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label-col', default='verify_consensus')
    ap.add_argument('--limit', type=int, default=0, help='0 = all plots in EDGE_DIR')
    a = ap.parse_args()
    man = pd.read_csv(f'{EDGE_DIR}/edge_case_manifest.csv')
    label_col = a.label_col if a.label_col in man.columns else 'verify_claude'
    print(f"few-shot label column: {label_col}")
    fewshot = build_fewshot(man, label_col)
    if not fewshot:
        print("No verified exemplars found — fill the verify_* columns first."); return

    plots = sorted(glob.glob(f'{EDGE_DIR}/trade_*.png'))
    if a.limit:
        plots = plots[:a.limit]
    rows = []
    for i, p in enumerate(plots):
        msgs = [{"role": "system", "content": SYSTEM}] + fewshot + \
               [{"role": "user", "content": [{"type": "text", "text": "Classify this trade:"}, img_content(p)]}]
        try:
            res = json.loads(call(msgs))
        except Exception as e:
            res = {"archetype": "ERR", "flag": str(e)[:80]}
        res['plot'] = os.path.basename(p)
        rows.append(res)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(plots)}")
    out = pd.DataFrame(rows)
    out.to_csv(f'{EDGE_DIR}/gemma_triage_results.csv', index=False)

    # calibration: agreement with the verified archetype on the labeled set
    j = out.merge(man[['plot', 'arch']], on='plot', how='left')
    agree = (j['archetype'] == j['arch']).mean() * 100
    print(f"\nGemma archetype agreement vs verified labels: {agree:.0f}%  (n={j['arch'].notna().sum()})")
    print(out['archetype'].value_counts().to_string())
    print(f"[results -> {EDGE_DIR}/gemma_triage_results.csv]")


if __name__ == '__main__':
    main()
