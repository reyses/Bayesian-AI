"""Calibrate a local vision LLM (via Ollama OpenAI-compatible) against the 3-way consensus
labels before trusting it to triage the full trade set. Shape-based rubric + one-word output
(more robust for smaller VLMs than strict JSON). Reports agreement % + confusion.

Run (after the model is pulled):
  python research/gemma_calibrate.py --model gemma4 [--fewshot] [--n 0]
"""
import os, glob, base64, json, urllib.request, argparse, collections
import pandas as pd

D = 'reports/findings/edge_cases_clean'
ARCHES = ['CLEAN_RIDE', 'GAVE_BACK', 'STOPPED', 'CHOP', 'GAP_TRUNCATED', 'SMALL_WIN', 'SMALL_LOSS']
RUBRIC = ("Signed-PnL trade chart (green=entry@0, red=exit, orange star=peak). Classify the SHAPE:\n"
 "CLEAN_RIDE=rises and STAYS up near peak. GAVE_BACK=rises to a high peak then FALLS back toward/below 0. "
 "STOPPED=drops fast and stays down (~-50). CHOP=flat, small wiggles near 0. "
 "GAP_TRUNCATED=only a few points then abrupt end. SMALL_WIN=modest positive end. SMALL_LOSS=modest negative end.\n"
 "Reply EXACTLY ONE of: " + ", ".join(ARCHES))


def b64(p): return base64.b64encode(open(p, 'rb').read()).decode()
def img(p): return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(p)}"}}


def call(model, msgs):
    body = json.dumps({"model": model, "temperature": 0, "messages": msgs}).encode()
    r = urllib.request.Request("http://localhost:11434/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"})
    t = json.loads(urllib.request.urlopen(r, timeout=240).read())['choices'][0]['message']['content'].upper().replace(' ', '_')
    for a in ARCHES:
        if a in t:
            return a
    return t[:15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--fewshot', action='store_true')
    ap.add_argument('--n', type=int, default=0, help='per-class sample (0=all)')
    a = ap.parse_args()
    m = pd.read_csv(f'{D}/edge_case_manifest.csv')
    m['p'] = D + '/' + m['plot']; m = m[m['p'].apply(os.path.exists)]

    pre = []
    if a.fewshot:
        ex = {x: m[m['verify_consensus'] == x].iloc[0]['p'] for x in ['CLEAN_RIDE', 'GAVE_BACK', 'STOPPED'] if len(m[m['verify_consensus'] == x])}
        c = [{"type": "text", "text": "Labeled examples:"}]
        for x, p in ex.items():
            c += [{"type": "text", "text": f"= {x}:"}, img(p)]
        pre = [{"role": "user", "content": c}]
        m = m[~m['p'].isin(ex.values())]

    q = m if a.n == 0 else m.groupby('verify_consensus').head(a.n)
    ok = 0; n = 0; conf = collections.Counter()
    for _, r in q.iterrows():
        msgs = pre + [{"role": "user", "content": [{"type": "text", "text": RUBRIC}, img(r['p'])]}]
        g = call(a.model, msgs); c = r['verify_consensus']; n += 1; ok += (g == c)
        conf[(c, g)] += 1
        print(f"{'OK ' if g==c else '.. '} {c:13s} -> {g}")
    print(f"\n{a.model} {'few-shot' if a.fewshot else 'zero-shot'} AGREEMENT: {ok}/{n} = {ok/n*100:.0f}%  (random ~14%)")
    print("verdict:", "USABLE (>=70%)" if ok/n >= 0.7 else "MARGINAL" if ok/n >= 0.5 else "NOT reliable")


if __name__ == '__main__':
    main()
