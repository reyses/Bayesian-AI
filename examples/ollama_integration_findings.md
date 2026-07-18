# Hermes Agent Ollama Integration Findings

After reviewing the `feat/ollama-desktop-integration` branch in the cloned Hermes Agent repository, here are the key findings on how they robustly integrate the Ollama HTTP API, specifically regarding context window management.

## 1. Context Window Calculation (`query_ollama_num_ctx`)
Ollama has a common pitfall where it defaults to a very low context window (e.g. 2048) unless explicitly overridden. Furthermore, you cannot set the context length through the standard OpenAI-compatible API (`/v1/chat/completions`); it must be configured server-side.

To work around this, the Hermes Agent codebase implements a custom context probe function (`query_ollama_num_ctx` located in `agent/model_metadata.py`). 

Here is how the probing logic operates:
1. **Hits the `/api/show` endpoint**: It bypasses the standard OpenAI `/v1/models` endpoint and uses Ollama's native `/api/show` endpoint by passing the model name.
2. **Checks Modelfile (`parameters` -> `num_ctx`)**: It first parses the `parameters` block returned by `/api/show` and looks for an explicit `num_ctx` definition. This ensures that if a user has explicitly clamped or raised the context window via a Modelfile, Hermes honors it.
3. **Falls back to GGUF Training Max (`model_info.*.context_length`)**: If no `num_ctx` parameter is specified in the Modelfile, it inspects the `model_info` block and extracts the `context_length` parameter, representing the absolute maximum trained context for the model.

## 2. API Caching
To avoid latency overhead, the result of this context probe (`_query_ollama_api_show_uncached`) is heavily cached. It memoizes the result keyed by `("ollama_show", bare_model_name, base_url)` with a TTL, guaranteeing that the pipeline does not hit the `/api/show` endpoint for every single chat inference call.

## 3. URL Translation and WSL Bridging
There are multiple checks in `tests/test_base_url_hostname.py` showing that they are rigorously enforcing separation between `ollama.com` (Ollama Cloud) and local Ollama (`127.0.0.1` / `localhost`). If you are running the `dojo_forge` in WSL but Ollama is running on the Windows host, remember to:
- Bind Ollama to `0.0.0.0` on Windows by setting the `OLLAMA_HOST` environment variable.
- Make your WSL scripts point to the Windows host IP (or `$WSL_HOST`) rather than `localhost`.

## Applying this to `dojo_forge`
If `run_episode_ollama` in `forge_harness.py` is experiencing silent truncation or dropping context, you should port a simplified version of `query_ollama_num_ctx`. Before starting an episode, do an HTTP `POST` to `http://localhost:11434/api/show` with `{"model": "gemma4:e2b"}`, parse out the context limits, and ensure your payload sizes are managed accordingly.
