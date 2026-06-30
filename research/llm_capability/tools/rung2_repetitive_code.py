"""RUNG 2 — repetitive CODE implementation under house rules.

Task: generate boilerplate that follows THIS repo's conventions (no magic numbers; every
constant a named, commented field). We give N specs; for each we (a) check the reply parses
as valid Python (py_compile via compile()), and (b) check it obeys the no-magic-numbers rule
heuristically. Objective pass/fail per spec — measures code-gen reliability, not taste.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ollama_client import chat, TEXT_MODEL  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORT = os.path.join(ROOT, "research", "llm_capability", "reports", "rung2_repetitive_code.md")

SYSTEM = (
    "You write Python for a trading codebase with a STRICT rule: NO magic numbers. Every numeric "
    "constant must be a named module-level constant with an inline comment explaining its origin "
    "(only pi/e exempt). Reply with ONLY a fenced ```python code block, no prose."
)

SPECS = [
    ("dataclass field",
     "Add three fields to a TradingConfig dataclass: z_entry threshold (1.8481), z_exit threshold "
     "(0.4752), and tick_value in dollars (0.50 for MNQ). Use named constants, not literals in the field defaults."),
    ("converter fn",
     "Write a function points_to_dollars(points, contracts) for MNQ where one point = 4 ticks and "
     "one tick = $0.50. No magic numbers."),
    ("enum mapping",
     "Write a function action_to_delta(action: str) -> int mapping 'LONG'->+1, 'SHORT'->-1, "
     "'FLAT'->0, raising ValueError otherwise. Use a named dict constant."),
    ("rolling stat",
     "Write a function rolling_zscore(series, window) returning the z-score of the last point over "
     "a trailing window, guarding zero std with a small epsilon named constant."),
]


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    return (m.group(1) if m else text or "").strip()


def has_magic_number(code):
    """Heuristic: a numeric literal used inline in logic (not in a NAMED = assignment / not 0/1/-1)."""
    # strip comments + strings
    nostr = re.sub(r"#.*", "", code)
    nostr = re.sub(r"(['\"]).*?\1", "", nostr)
    for ln in nostr.splitlines():
        # allow named-constant assignment lines: NAME = 1.23  (the sanctioned place for a literal)
        if re.match(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*[:=]", ln) and "=" in ln and "==" not in ln:
            continue
        for num in re.findall(r"(?<![\w.])\d+\.?\d*", ln):
            if num not in ("0", "1", "2"):  # 0/1/2 are structural (indexing, sign), tolerated
                return True, num
    return False, None


def main():
    lines, compiles, clean, lat = [], 0, 0, []
    def w(s):
        print(s); lines.append(s)
    w(f"# RUNG 2 — repetitive code under no-magic-numbers rule | model={TEXT_MODEL} | n={len(SPECS)}\n")
    w(f"{'spec':>16} | {'compiles':>8} | {'no-magic':>8} | note")
    w("-" * 64)
    for name, spec in SPECS:
        reply, dt = chat(spec, system=SYSTEM, num_predict=400)
        lat.append(dt)
        code = extract_code(reply)
        ok_compile = False
        try:
            compile(code, "<gen>", "exec")
            ok_compile = True
            compiles += 1
        except SyntaxError as e:
            note = f"SyntaxError: {str(e)[:30]}"
        magic, num = has_magic_number(code)
        if ok_compile:
            if not magic:
                clean += 1
                note = "clean"
            else:
                note = f"magic literal {num!r}"
        w(f"{name:>16} | {'Y' if ok_compile else 'n':>8} | "
          f"{('Y' if not magic else 'n') if ok_compile else '-':>8} | {note}")
    n = len(SPECS)
    w("")
    w(f"COMPILES: {compiles}/{n} = {compiles/n:.0%}")
    w(f"OBEYS no-magic-numbers (of those that compile): {clean}/{compiles if compiles else 1}")
    w(f"latency/call: mean {sum(lat)/len(lat):.1f}s")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
