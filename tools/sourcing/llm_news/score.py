"""LLM news-intensity scorer using local Llama-3.1-8B-Instruct (Q4_K_M GGUF).

Reads all press release .txt files in DATA/CROSS_DAY/raw/press_releases/,
plus the _metadata.json sidecar written by fetch.py. For each release,
asks the LLM for a single integer intensity score (0-10) and a one-sentence
rationale. Output is deterministic (temperature=0, top_k=1, fixed seed).

Anti-cheating measures:
  1. Prompt explicitly instructs LLM to score based ONLY on the text, not
     on knowledge of subsequent market reactions.
  2. CLI flag --test-synthetic runs a hawkish-vs-dovish synthetic statement
     pair through the model. Hawkish must score higher than dovish, else
     the prompt is bugged and we abort before scoring real releases.

Model path resolution order:
  1. $BAYESIAN_LLM_GGUF_PATH environment variable (override)
  2. models/llama-3.1-8b-instruct-q4_k_m.gguf (default)

To download the model (one time, ~4.7GB):
  huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
      Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
      --local-dir models --local-dir-use-symlinks False
  mv models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf models/llama-3.1-8b-instruct-q4_k_m.gguf

Or any equivalent Q4_K_M GGUF of Llama-3.1-8B-Instruct. If you want to
upgrade to a larger model (Qwen2.5-14B, Mistral-Small-24B), set
BAYESIAN_LLM_GGUF_PATH to the new file -- everything else stays the same.

Output: DATA/CROSS_DAY/dev/news_scores_v1.parquet
  columns: date, event_type, release_ts_et, intensity, rationale,
           model_id, prompt_hash
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PR_DIR = Path('DATA/CROSS_DAY/raw/press_releases')
META_IN = RAW_PR_DIR / '_metadata.json'
OUT_PATH = Path('DATA/CROSS_DAY/dev/news_scores_v1.parquet')

DEFAULT_MODEL_PATH = Path('models/llama-3.1-8b-instruct-q4_k_m.gguf')
SEED = 42
N_CTX = 4096
MAX_TOKENS = 200

PROMPT_TEMPLATE = """You are scoring a US economic press release for short-term market intensity.

Score ONLY based on the language of the text below. Do not use any knowledge of subsequent market reactions, prior knowledge of how this release was received, or any external context. Treat the text as an isolated document.

Output a single JSON object with two fields and nothing else:
  {{"intensity": <integer 0-10>, "rationale": "<one short sentence>"}}

Scoring guide:
  0-2 = boilerplate / on-consensus / no surprise (e.g. "policy unchanged, language unchanged")
  3-5 = some new information but expected direction (e.g. typical FOMC tweak)
  6-8 = clear surprise vs. consensus / significant policy shift (e.g. unexpected guidance change, large CPI/NFP miss)
  9-10 = major shock (e.g. unscheduled rate move, employment crash, inflation print >2 sigma off consensus)

EVENT_TYPE: {event_type}
RELEASE_DATE: {date}

TEXT:
{text}

Output JSON only, no preamble:"""


def _prompt_hash() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode('utf-8')).hexdigest()[:12]


def _resolve_model_path() -> Path:
    override = os.environ.get('BAYESIAN_LLM_GGUF_PATH')
    p = Path(override) if override else DEFAULT_MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f'GGUF model not found at {p}.\n'
            f'Download Llama-3.1-8B-Instruct Q4_K_M and place it at the default path,\n'
            f'or set BAYESIAN_LLM_GGUF_PATH to point to your local GGUF file.\n'
            f'See tools/sourcing/llm_news/score.py docstring for download instructions.'
        )
    return p


def _load_llama():
    from llama_cpp import Llama
    model_path = _resolve_model_path()
    return Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=N_CTX,
        seed=SEED,
        verbose=False,
    ), model_path.name


def _parse_json_score(text: str) -> tuple[int | None, str | None]:
    """Extract {"intensity": int, "rationale": str} from model output."""
    m = re.search(r'\{[^{}]*"intensity"\s*:\s*(-?\d+)[^{}]*\}', text, re.DOTALL)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return int(m.group(1)) if m.group(1).lstrip('-').isdigit() else None, None
    intensity = obj.get('intensity')
    rationale = obj.get('rationale', '')
    if not isinstance(intensity, (int, float)):
        return None, None
    return int(np.clip(int(intensity), 0, 10)), str(rationale)[:280]


def _score_one(llm, model_id: str, event_type: str, date_str: str, text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(event_type=event_type.upper(),
                                     date=date_str, text=text[:12000])
    result = llm.create_chat_completion(
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=SEED,
        max_tokens=MAX_TOKENS,
    )
    raw = result['choices'][0]['message']['content']
    intensity, rationale = _parse_json_score(raw)
    if intensity is None:
        result2 = llm.create_chat_completion(
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            top_k=1,
            top_p=1.0,
            seed=SEED + 1,
            max_tokens=MAX_TOKENS,
        )
        raw2 = result2['choices'][0]['message']['content']
        intensity, rationale = _parse_json_score(raw2)
        if intensity is None:
            intensity = 0
            rationale = f'[parse_fail; raw={raw[:120]}]'
    return {
        'intensity': intensity,
        'rationale': rationale or '',
        'model_id': model_id,
        'prompt_hash': _prompt_hash(),
        'raw_excerpt': raw[:240],
    }


def test_synthetic(llm=None, model_id: str | None = None) -> bool:
    """Run hawkish-vs-dovish synthetic statement test.
    Returns True iff hawkish > dovish. Aborts caller on False.

    If `llm` and `model_id` are not provided, loads its own instance. Pass
    a pre-loaded llm to avoid double-loading the GGUF (~30s on RTX 4080)."""
    HAWKISH = ('The Committee has determined that an additional 50-basis-point '
               'increase in the target range is appropriate to bring inflation '
               'firmly back to the 2 percent objective. Risks to the inflation '
               'outlook remain skewed materially to the upside, and the Committee '
               'is prepared to act forcefully should price pressures fail to abate. '
               'The Committee anticipates further substantial tightening will be needed.')
    DOVISH = ('In light of the cumulative tightening of monetary policy and the '
              'lags with which policy affects economic activity and inflation, the '
              'Committee has decided to leave the target range unchanged and judges '
              'that the risks to the inflation outlook are now more balanced. The '
              'Committee will assess incoming data carefully and stands ready to '
              'ease policy should economic conditions warrant.')

    if llm is None:
        print('Loading LLM for synthetic anti-cheating test...')
        llm, model_id = _load_llama()
        print(f'  model: {model_id}')
        print()

    print('Scoring HAWKISH synthetic statement...')
    h = _score_one(llm, model_id, 'fomc', '2099-01-01', HAWKISH)
    print(f'  intensity={h["intensity"]}  rationale={h["rationale"]!r}')

    print('Scoring DOVISH synthetic statement...')
    d = _score_one(llm, model_id, 'fomc', '2099-01-02', DOVISH)
    print(f'  intensity={d["intensity"]}  rationale={d["rationale"]!r}')

    passed = h['intensity'] > d['intensity']
    print()
    print(f'Hawkish ({h["intensity"]}) > Dovish ({d["intensity"]}): {passed}')
    if not passed:
        print('FAIL: anti-cheating synthetic test did not produce ordered scores.')
        print('      The prompt is bugged or the model is too small. DO NOT proceed.')
    else:
        print('PASS: ordering correct. Proceed to scoring real releases.')
    return passed


def main(skip_synthetic: bool = False) -> dict:
    if not META_IN.exists():
        raise FileNotFoundError(
            f'{META_IN} missing. Run `python -m tools.sourcing.llm_news.cli fetch` first.'
        )

    metadata = json.loads(META_IN.read_text(encoding='utf-8'))
    print(f'Loading LLM (one-time)...')
    llm, model_id = _load_llama()
    print(f'  model: {model_id}')
    print()

    if not skip_synthetic:
        ok = test_synthetic(llm, model_id)
        if not ok:
            raise RuntimeError(
                'Synthetic anti-cheating test failed; aborting before real scoring. '
                'Iterate on prompt or switch model, then re-run.'
            )

    print()
    print(f'Backfill: {len(metadata)} releases to score')
    print()

    rows = []
    for i, m in enumerate(metadata):
        path = Path(m['path'])
        if not path.exists():
            print(f'  [{i+1}/{len(metadata)}] MISSING {path}, skipping')
            continue
        text = path.read_text(encoding='utf-8')
        if len(text) < 100:
            print(f'  [{i+1}/{len(metadata)}] TOO_SHORT {path} ({len(text)} chars), skipping')
            continue
        date_str = m['date']
        event_type = m['event_type']
        out = _score_one(llm, model_id, event_type, date_str, text)
        ts_et = pd.Timestamp(m['release_ts_et'])  # tz-aware
        print(f'  [{i+1}/{len(metadata)}] {date_str} {event_type:4s}  '
              f'intensity={out["intensity"]}  '
              f'rationale={out["rationale"][:80]!r}')
        rows.append({
            'date': pd.Timestamp(date_str).date(),
            'event_type': event_type,
            'release_ts_et': ts_et,
            'intensity': out['intensity'],
            'rationale': out['rationale'],
            'model_id': out['model_id'],
            'prompt_hash': out['prompt_hash'],
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        std_int = float(df['intensity'].std())
        print(f'\nScore distribution: mean={df["intensity"].mean():.2f}  '
              f'std={std_int:.2f}  min={df["intensity"].min()}  '
              f'max={df["intensity"].max()}')
        if std_int < 1.5:
            print(f'WARN: std(intensity)={std_int:.2f} < 1.5 -- LLM may be producing flat scores. '
                  f'Iterate prompt or upgrade model before retraining DRS.')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f'\nWrote: {OUT_PATH}  ({len(df)} rows)')
    return {'n_scored': len(df), 'out_path': str(OUT_PATH)}


if __name__ == '__main__':
    main()
