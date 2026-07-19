"""
gpu_llama_bench.py — CUDA-accelerated llama-cpp-python sanity + benchmark for qwen3:14b.

Loads the qwen3:14b Q4 GGUF (ollama blob) with GPU offload, proves CUDA is actually
used (nvidia-smi VRAM delta + llama.cpp offload log), runs the EXIT/HOLD logprobs
readout used by the dojo forge, and benchmarks cold load + warm ~6000-token evals.

Run (from the dedicated CUDA venv):
    C:\\Users\\reyse\\venvs\\llamacpp-cuda\\Scripts\\python.exe research\\dojo_forge\\tools\\gpu_llama_bench.py

Writes: research/dojo_forge/reports/gpu_llama_bench.md
"""
import os
import sys
import time
import math
import json
import site
import subprocess
from pathlib import Path

import numpy as np

# --- 1. Make the venv-bundled CUDA runtime DLLs discoverable BEFORE importing llama_cpp.
#     ggml-cuda.dll implicitly links cudart64_12.dll / cublas64_12.dll / nvrtc64_120_0.dll.
#     These ship in the nvidia-*-cu12 wheels under site-packages/nvidia/*/bin.
def _add_cuda_dll_dirs():
    added = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia = Path(sp) / "nvidia"
        if not nvidia.is_dir():
            continue
        for sub in ("cuda_runtime", "cublas", "cuda_nvrtc"):
            bindir = nvidia / sub / "bin"
            if bindir.is_dir():
                os.add_dll_directory(str(bindir))
                os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
                added.append(str(bindir))
    return added

DLL_DIRS = _add_cuda_dll_dirs()

from llama_cpp import Llama  # noqa: E402
import llama_cpp  # noqa: E402

MODEL_PATH = r"D:\ollama\models\blobs\sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e"
N_CTX = 8192
REPORT = Path(__file__).resolve().parents[1] / "reports" / "gpu_llama_bench.md"


def nvidia_smi_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=15,
        ).decode().strip().splitlines()
        return int(out[0].strip())
    except Exception as e:
        return f"err:{e}"


def _first_token_id(llm, s):
    """First token id of a candidate answer string (no BOS, no special)."""
    ids = llm.tokenize(s.encode("utf-8"), add_bos=False, special=False)
    return ids[0]


def extract_exit_hold(llm, prompt_text, exit_id, hold_id):
    """Low-level readout: eval the prompt, read the LAST-token logit vector directly
    (llm.scores[0] when logits_all=False), pull the raw logits for the EXIT/HOLD token
    ids. Memory-cheap (no all-position logits buffer) and matches the dojo readout."""
    llm.reset()
    toks = llm.tokenize(prompt_text.encode("utf-8"), add_bos=False, special=True)
    t0 = time.time()
    llm.eval(toks)
    logits = np.asarray(llm.scores[0, :], dtype=np.float64)  # last-token logits
    dur = time.time() - t0

    l_exit = float(logits[exit_id])
    l_hold = float(logits[hold_id])
    p_exit = math.exp(l_exit) / (math.exp(l_exit) + math.exp(l_hold))

    # Rank check: is EXIT/HOLD in the top-50 tokens by logit? (honors the "not floored" test)
    top50_idx = np.argpartition(-logits, 50)[:50]
    top50_idx = top50_idx[np.argsort(-logits[top50_idx])]
    argmax_id = int(top50_idx[0])
    argmax_piece = llm.detokenize([argmax_id]).decode("utf-8", "ignore")
    top = {}
    for tid in top50_idx[:12]:
        piece = llm.detokenize([int(tid)]).decode("utf-8", "ignore")
        top[piece.encode("ascii", "ignore").decode("ascii")] = round(float(logits[tid]), 3)
    return {
        "dur": dur, "lp_exit": l_exit, "lp_hold": l_hold, "p_exit": p_exit,
        "top": top, "argmax": argmax_piece,
        "exit_present": exit_id in set(int(i) for i in top50_idx),
        "hold_present": hold_id in set(int(i) for i in top50_idx),
        "n_tokens": len(toks),
    }


def build_long_prompt(llm, target_tokens=6000):
    """Build a ~target_tokens chat prompt ending in a closed </think> trace, then
    ask for the next action. Designed so EXIT is near-certain (price collapsed)."""
    head = (
        "<|im_start|>system\nYou are an expert MNQ futures exit trader. Each frame gives "
        "market state during an open long position. Respond with exactly one token: EXIT to "
        "close the position, or HOLD to stay in.<|im_end|>\n"
    )
    # Filler frames to reach the token budget (a realistic-length episode context).
    filler = ""
    frame_id = 0
    while True:
        frame_id += 1
        price = 100.0 - frame_id * 0.05
        filler += (
            f"<|im_start|>user\nFrame {frame_id}: price={price:.2f} z=+0.4 lambda=+0.02 "
            f"vol=normal ride_len={frame_id} bars. Action:<|im_end|>\n"
            f"<|im_start|>assistant\n<think>Position still running with the trend. Hold.</think>\nHOLD<|im_end|>\n"
        )
        toks = llm.tokenize((head + filler).encode("utf-8"), add_bos=True, special=True)
        if len(toks) >= target_tokens:
            break
        if frame_id > 4000:
            break
    # Final decisive frame: price has collapsed, the <think> concludes EXIT.
    final = (
        "<|im_start|>user\nFrame FINAL: price=61.30 z=-4.8 lambda=-0.31 vol=SPIKE "
        "drawdown=-38pts trend BROKEN, snapping back hard against the position. Action:<|im_end|>\n"
        "<|im_start|>assistant\n<think>The trend has clearly broken. Price collapsed 38 points "
        "against the position with a volatility spike and lambda strongly negative. The run is over. "
        "I will EXIT now.</think>\n"
    )
    prompt = head + filler + final
    ntok = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))
    return prompt, ntok


def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
         "--format=csv,noheader"], timeout=15).decode().strip()

    log(f"llama_cpp version: {llama_cpp.__version__}")
    log(f"python: {sys.version.split()[0]}  exe: {sys.executable}")
    log(f"nvidia-smi: {driver}")
    log(f"CUDA DLL dirs added: {DLL_DIRS}")

    vram_before = nvidia_smi_used_mib()
    log(f"VRAM used before load: {vram_before} MiB")

    # Try full offload, fall back on OOM.
    offload_result = None
    for ngl in (-1, 40, 35, 30):
        try:
            t0 = time.time()
            llm = Llama(
                model_path=MODEL_PATH, n_gpu_layers=ngl, n_ctx=N_CTX,
                seed=42, logits_all=False, verbose=True,
            )
            cold = time.time() - t0
            offload_result = (ngl, cold, llm)
            log(f"\nLoaded with n_gpu_layers={ngl} in {cold:.2f}s cold")
            break
        except Exception as e:
            log(f"n_gpu_layers={ngl} FAILED: {type(e).__name__}: {e}")

    if offload_result is None:
        log("\nALL offload attempts failed. See errors above.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        return

    ngl, cold, llm = offload_result
    vram_after = nvidia_smi_used_mib()
    log(f"VRAM used after load: {vram_after} MiB")
    try:
        delta = vram_after - vram_before
        log(f"VRAM delta (model on GPU): {delta} MiB")
    except Exception:
        delta = "n/a"

    exit_id = _first_token_id(llm, "EXIT")
    hold_id = _first_token_id(llm, "HOLD")
    log(f"token ids -> EXIT={exit_id} ('{llm.detokenize([exit_id]).decode('utf-8','ignore')}')  "
        f"HOLD={hold_id} ('{llm.detokenize([hold_id]).decode('utf-8','ignore')}')")

    # --- Sanity: EXIT/HOLD logits on a near-certain EXIT frame ---
    log("\n=== SANITY: near-certain EXIT frame ===")
    prompt, ntok = build_long_prompt(llm, target_tokens=6000)
    log(f"benchmark prompt length: {ntok} tokens")
    san = extract_exit_hold(llm, prompt, exit_id, hold_id)
    log(f"top-12 tokens by logit: {san['top']}")
    log(f"argmax next token: '{san['argmax']}'")
    log(f"EXIT logit: {san['lp_exit']:.4f}  in_top50={san['exit_present']}")
    log(f"HOLD logit: {san['lp_hold']:.4f}  in_top50={san['hold_present']}")
    log(f"P(EXIT) = {san['p_exit']:.6f}")
    sanity_ok = san["exit_present"] and san["hold_present"]
    log(f"SANITY {'PASS' if sanity_ok else 'FAIL'}: both EXIT and HOLD in top-50 by logit")

    # --- Warm benchmark: 5 evals of the ~6000-token prompt ---
    log("\n=== WARM BENCHMARK: 5x ~6000-token evals ===")
    times = []
    for i in range(5):
        r = extract_exit_hold(llm, prompt, exit_id, hold_id)  # resets + reprocesses full prompt
        times.append(r["dur"])
        log(f"  eval {i+1}: {r['dur']:.3f}s  P(EXIT)={r['p_exit']:.4f}")
    times.sort()
    p50 = times[len(times)//2]
    p95 = times[int(len(times)*0.95) - 1] if len(times) >= 2 else times[-1]
    log(f"\nwarm p50: {p50:.3f}s   warm p95: {p95:.3f}s   min: {times[0]:.3f}s   max: {times[-1]:.3f}s")

    vram_run = nvidia_smi_used_mib()
    log(f"VRAM used during run: {vram_run} MiB  (headroom to 12288 MiB: {12288 - vram_run if isinstance(vram_run,int) else 'n/a'} MiB)")

    # --- Write markdown report ---
    md = []
    md.append("# GPU llama-cpp-python benchmark — qwen3:14b (Q4 GGUF)\n")
    md.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    md.append("## Result\n")
    md.append(f"- **GPU working: {'YES' if isinstance(delta,int) and delta > 3000 else 'CHECK'}** "
              f"(VRAM delta {delta} MiB on load)\n")
    md.append(f"- Wheel: `llama_cpp_python-0.3.43+cu124-cp311-cp311-win_amd64.whl` (JamePeng fork)\n")
    md.append(f"- llama_cpp version: {llama_cpp.__version__}\n")
    md.append(f"- venv: `C:\\Users\\reyse\\venvs\\llamacpp-cuda`\n")
    md.append(f"- offload: n_gpu_layers={ngl}, n_ctx={N_CTX}\n")
    md.append(f"- driver: {driver}\n")
    md.append("\n## DLL mechanism\n")
    md.append("`ggml-cuda.dll` implicitly links `cudart64_12.dll` / `cublas64_12.dll` / "
              "`nvrtc64_120_0.dll`, supplied by the `nvidia-*-cu12` wheels in the same venv. "
              "The script calls `os.add_dll_directory()` on each `site-packages/nvidia/*/bin` "
              "before importing `llama_cpp`. Dirs added:\n")
    for d in DLL_DIRS:
        md.append(f"- `{d}`\n")
    md.append("\n## Timings\n")
    md.append("| metric | seconds |\n|---|---|\n")
    md.append(f"| cold load ({ntok}-tok not incl.) | {cold:.2f} |\n")
    md.append(f"| warm p50 (~{ntok}-tok eval) | {p50:.3f} |\n")
    md.append(f"| warm p95 | {p95:.3f} |\n")
    md.append(f"| warm min | {times[0]:.3f} |\n")
    md.append(f"| warm max | {times[-1]:.3f} |\n")
    md.append(f"\nCPU baseline (0.3.34 wheel): 28s cold, 1-4s warm/frame.\n")
    md.append("\n## VRAM\n")
    md.append(f"- before load: {vram_before} MiB\n- after load: {vram_after} MiB\n")
    md.append(f"- during run: {vram_run} MiB\n- headroom to 12288 MiB at ctx {N_CTX}: "
              f"{12288 - vram_run if isinstance(vram_run,int) else 'n/a'} MiB\n")
    md.append("\n## Sanity (EXIT/HOLD readout)\n")
    md.append("Low-level readout: `llm.eval(tokens)` then read the last-token logit vector "
              "`llm.scores[0]` directly (works with `logits_all=False`, no 3.7 GB all-position "
              "logits buffer — that is what OOMs a 12 GB card at 6 k tokens).\n")
    md.append(f"- benchmark prompt: {ntok} tokens, ends with closed `</think>` concluding EXIT\n")
    md.append(f"- EXIT/HOLD token ids: {exit_id} / {hold_id}\n")
    md.append(f"- argmax next token: `{san['argmax']}`\n")
    md.append(f"- EXIT logit: {san['lp_exit']:.4f} (in top-50 = {san['exit_present']})\n")
    md.append(f"- HOLD logit: {san['lp_hold']:.4f} (in top-50 = {san['hold_present']})\n")
    md.append(f"- P(EXIT) = {san['p_exit']:.6f}\n")
    md.append(f"- top-12 tokens by logit: `{san['top']}`\n")
    md.append(f"- **SANITY {'PASS' if sanity_ok else 'FAIL'}**\n")
    md.append("\n## Reproduction\n")
    md.append("```powershell\n")
    md.append("uv venv C:\\Users\\reyse\\venvs\\llamacpp-cuda --python 3.11\n")
    md.append("$VP = 'C:\\Users\\reyse\\venvs\\llamacpp-cuda\\Scripts\\python.exe'\n")
    md.append("uv pip install --python $VP --link-mode=copy `\n")
    md.append("  'https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.43-cu124-win-20260718/llama_cpp_python-0.3.43%2Bcu124-cp311-cp311-win_amd64.whl' `\n")
    md.append("  numpy diskcache jinja2 typing_extensions nvidia-cuda-runtime-cu12 nvidia-cublas-cu12\n")
    md.append("& $VP research\\dojo_forge\\tools\\gpu_llama_bench.py\n")
    md.append("```\n")
    md.append("\n## Switch instruction for AG\n")
    md.append("Point the batch runner at `C:\\Users\\reyse\\venvs\\llamacpp-cuda\\Scripts\\python.exe`, "
              "keep the `os.add_dll_directory` block (copy `_add_cuda_dll_dirs()` from this script "
              "to the top of the runner, before `from llama_cpp import Llama`), and load the model "
              "with `n_gpu_layers=-1`.\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(md), encoding="utf-8")
    log(f"\nReport written: {REPORT}")


if __name__ == "__main__":
    main()
