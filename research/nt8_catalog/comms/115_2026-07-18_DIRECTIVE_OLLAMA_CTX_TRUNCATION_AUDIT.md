# DIRECTIVE — ollama context-truncation audit (gen-0 integrity) + ctx discipline
**Doc:** 115 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **For:** AG
**Source:** `examples/ollama_integration_findings.md` — from the SEPARATE
Hermes-fork investigation (direct study of the upstream repo's
`feat/ollama-desktop-integration` branch). Credit: that track, not AG/Claude.

## The risk (why this is PRIORITY over finishing gen-0)
Ollama silently clips prompts to the effective `num_ctx` (default 2048; AG's
running instance showed 4096). Forge episodes = genome prefix + up to ~45
accumulated frames — the overflow victim is the EARLY context, i.e. **the
genome itself silently falls out of the window late in episodes**. That means
the measured policy stops being the written policy exactly on long trades,
with zero errors raised. Same failure class as the lookahead family: an
invisible integrity bug producing plausible results.

## Required of AG (in order)
1. **Probe**: port a simplified `query_ollama_num_ctx` (native `/api/show`,
   parse Modelfile `num_ctx` else `model_info.context_length`; cache it).
2. **Set**: every native `/api/chat` call passes `options.num_ctx` explicitly
   (the native endpoint accepts it; the OpenAI shim does not). Choose a value
   ≥ worst-case episode prompt (measure it; named constant).
3. **AUDIT the already-played gen-0 episodes**: for each, compare ollama's
   returned `prompt_eval_count` against the true prompt token size on the
   longest frames. Any episode whose prompt exceeded the effective window =
   **CONTAMINATED**: report the count, keep the tainted transcripts (labeled),
   re-run those episodes with the fix. Same eids; both transcripts retained.
4. **Assert forward**: hard-fail any request whose prompt would exceed the
   window. Loud failure, never silent clipping.
5. **WSL bridging** (native lane): `localhost` in WSL does not reach
   Windows-host ollama — `OLLAMA_HOST` binding / host IP per the findings doc.

## Also standing (from the mailbox, now on the record)
- Ollama model store migrates C:→D: at AG's next gate STOP (script staged);
  blob paths re-derived afterward, never hardcoded.
- Executor roster: gen-0 = gemma4:e2b; qwen3:14b enters gen-1 (think:false);
  deepseek-r1:14b = slow-arm/distiller only. The observed F2_QWEN_NATIVE lane
  must be explained in AG's next report (if the gemma blob fails under
  llama.cpp, paste the load error — a sanctioned reroute needs its receipt).
- Doc-114 acceptance table for native evidence remains open.

AG's next report covers: truncation audit results, native-blob evidence (or
error), the qwen-lane explanation, gen-0 status. Number it 116.
