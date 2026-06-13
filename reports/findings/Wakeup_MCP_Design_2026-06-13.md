# Wake-Antigravity MCP Design Note
**Date:** 2026-06-13
**Author:** Gemini

## The Goal
The user requested a mechanism for external scripts to voluntarily trigger a "wakeup" signal to Gemini without polluting the chat prompt. The user recalled a `wakeup(1)` command that sent a signal.

## The Architecture Problem
MCP (Model Context Protocol) is strictly an **Agent → Server** paradigm. The agent queries tools on the server. External scripts *cannot* call an MCP tool to force the agent to do something, because the agent is the one initiating requests. If a script runs an MCP tool locally, the agent never sees it.

## The Antigravity Solution (Reactive Wakeups)
In Antigravity, the agent is connected to a messaging system that implements **Reactive Wakeup**. The system automatically resumes execution when a background task completes or sends a notification.

To create a voluntary, external wakeup trigger without API keys or exfiltration risks:
1. **The Watcher Task:** We launch a background task (e.g., via the `schedule` tool or `run_command` with a long loop) that simply watches a local file (e.g., `comms/WAKE_GEMINI.trigger`).
2. **The External Script:** When an external Python script finishes running (e.g., a heavy hyperparameter sweep), it writes to `comms/WAKE_GEMINI.trigger`.
3. **The Wakeup Event:** The watcher task detects the file modification and immediately prints a message to stdout. In Antigravity, this stdout message is instantly routed to the agent's context as a high-priority system message, immediately waking the agent from sleep.

## Implementation (Draft)
A draft MCP `gemini_wake_mcp.py` was created, but as noted, MCP is the wrong layer for inbound triggers. We leave it unregistered.

The correct implementation is to have the agent run a background file watcher task:
```bash
# Example background task command
while true; do
  inotifywait -e modify comms/WAKE_GEMINI.trigger
  echo "WAKEUP SIGNAL RECEIVED: External script completed."
done
```
*(On Windows, a PowerShell equivalent using `FileSystemWatcher` would be used).*

This ensures 0 API cost, instant latency, and complete security (no open ports or webhooks).
