"""GeminiDelegate MCP server — one-way delegation channel Claude -> Gemini.

Purpose: move INPUT-heavy / OUTPUT-light, verifiable, bounded work onto GOOGLE's quota so it
doesn't burn the Anthropic (Claude) session limit. Gemini reads the big files / parses the bulk
output and hands back a tight result; Claude keeps orchestration, decisions, and rigor.

Mirrors the proven telegram_mcp.py pattern: FastMCP + @mcp.tool + .env (resolved against this
script's dir) + raw REST via requests (no SDK version risk).

USE FOR: bulk read/summarize/extract, run-output parsing, first-pass code audit, large-file
scans. NOT FOR: rigor-critical decisions (firewall / CI / no-lookahead) — verify any result;
Gemini is a different model with different failure modes. ONE-WAY only (Claude calls Gemini and
gets a result) — no autonomous two-LLM chat loop (non-deterministic, can loop, wastes tokens).

Setup: put GEMINI_API_KEY=... in .env (same .env as the telegram server). Optional GEMINI_MODEL
(default gemini-2.0-flash — cheap/fast, which is the whole point for bulk reads).
"""
import os
import sys
from pathlib import Path
from typing import Optional
import requests
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

mcp = FastMCP("GeminiDelegate")

load_dotenv(Path(__file__).resolve().parent / ".env")
API_KEY = os.environ.get("GEMINI_API_KEY")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
MAX_INPUT_CHARS = 800_000   # ~200k-token guard so a runaway file can't blow the request


def _read_file(fp: str) -> str:
    p = fp if os.path.isabs(fp) else str(Path(__file__).resolve().parent / fp)
    if not os.path.exists(p):
        return f"\n[FILE NOT FOUND: {fp}]"
    try:
        return f"\n--- {fp} ---\n" + Path(p).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"\n[READ ERROR {fp}: {e}]"


@mcp.tool()
def gemini_delegate(task: str, files: Optional[list[str]] = None, model: str = "") -> str:
    """Delegate an INPUT-heavy / OUTPUT-light task to Gemini (Google's separate quota) to spare
    the Claude session limit. Reads the listed repo files (relative paths resolve against the
    repo root), sends task + file contents to Gemini, returns Gemini's text response.

    Args:
        task: the instruction for Gemini (be explicit; ask for a tight, structured result).
        files: optional list of file paths to include as context (repo-root-relative or absolute).
        model: optional Gemini model override (default gemini-2.0-flash).

    Returns Gemini's response text, or an error string. VERIFY the result — Gemini is a different
    model; treat its output like a subagent's (trust-but-check), never as a final decision.
    """
    if not API_KEY:
        return "ERROR: GEMINI_API_KEY not set in .env (next to gemini_mcp.py)."
    parts, total = [task], len(task)
    for fp in (files or []):
        chunk = _read_file(fp)
        if total + len(chunk) > MAX_INPUT_CHARS:
            chunk = chunk[: max(0, MAX_INPUT_CHARS - total)] + "\n[...TRUNCATED at input cap]"
            parts.append(chunk)
            break
        parts.append(chunk)
        total += len(chunk)
    prompt = "".join(parts)
    m = model or DEFAULT_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={API_KEY}"
    res = None
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=300)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        detail = (res.text[:500] if res is not None else "")
        return f"Gemini call failed ({m}): {e}\n{detail}"


if __name__ == "__main__":
    if not API_KEY:
        print("WARN: GEMINI_API_KEY not set in .env — gemini_delegate will return an error until set.",
              file=sys.stderr, flush=True)
    mcp.run()
