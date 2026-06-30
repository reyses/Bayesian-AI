"""Shared Ollama client for the local-LLM capability ladder.

All rungs call gemma4 through here so timing, determinism (temperature=0), and
format-failure accounting are consistent and centrally measured.

Ollama HTTP API (local, no network): POST /api/chat  {model, messages, options}.
Vision: attach base64 PNGs via messages[].images (gemma4:e2b is the multimodal variant).
"""
import base64
import json
import time
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
TEXT_MODEL = "gemma4:latest"   # 8B  — text / code / trading decisions
VISION_MODEL = "gemma4:e2b"    # 5.1B multimodal — chart reading


def chat(prompt, system=None, model=TEXT_MODEL, images=None, temperature=0.0,
         num_predict=512, timeout=180):
    """One causal call. Returns (text, latency_s). temperature=0 for reproducibility."""
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = [_b64(p) for p in images]
    body = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + [msg],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "seed": 0},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out.get("message", {}).get("content", ""), time.time() - t0


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_json(text):
    """Best-effort: pull the first {...} object out of a model reply. None on failure.
    LLMs wrap JSON in prose / code fences; this is the format-robustness layer whose
    failure rate we MEASURE (a documented limitation, not a silent fallback)."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
