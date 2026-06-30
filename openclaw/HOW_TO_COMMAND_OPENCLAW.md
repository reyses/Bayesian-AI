# How to Command OpenClaw

Reference for issuing commands/tasks to the local **OpenClaw** agent (v0.7.0+openclaw.2026.4.2).
Grounded in *this* install (`C:\Users\reyse\.openclaw\`). Living doc — add recipes as you learn them.

> ⚠️ **Config mismatch to fix first:** `~/.openclaw/openclaw.json` sets the model to **`ollama/llama3.3`**,
> but `ollama list` only has **gemma4** pulled. OpenClaw can't load a model it doesn't have. Either:
> - `ollama pull llama3.3`  (get the configured model), **or**
> - edit `openclaw.json` → `agents.defaults.model.primary` and `models.providers.ollama.models[].id`
>   to `gemma4:latest` (use what you have; vision-capable too).

---

## 1. What "commanding" OpenClaw means
OpenClaw is a **personal agent**, not a CLI you pass flags to. You command it four ways:

| mechanism | what it's for | where it lives |
|---|---|---|
| **Chat** (main session / control UI / channels) | direct, conversational commands | gateway on `127.0.0.1:18789` (token in `openclaw.json`); or a connected channel |
| **HEARTBEAT.md** | recurring background checks/tasks (batched, ~drifty timing) | `~/.openclaw/workspace/HEARTBEAT.md` |
| **Cron** | scheduled standalone tasks (exact timing, isolated, one-shots) | OpenClaw's cron (set via chat/UI/CLI) |
| **Skills** | give it new *tools/capabilities* | each skill's `SKILL.md` |

Plus two configuration layers that shape *how* it obeys:
- **`openclaw.json`** — gateway, auth, model/provider (Ollama), logging, updates.
- **`workspace/*.md`** — `SOUL.md` (who it is), `IDENTITY.md`, `USER.md` (who you are), `AGENTS.md` (operating rules), `TOOLS.md` (your local devices/hosts), `MEMORY.md` (long-term memory, main-session only).

---

## 2. Chat — the primary way to command it
- The agent runs behind a **local gateway**: `http://127.0.0.1:18789` (loopback, token auth — token is in `~/.openclaw/openclaw.json` under `gateway.auth.token`; **don't paste it into committed files**).
- Talk to it via the **control UI** or a connected **channel** (Discord/WhatsApp/etc., configured under `channels`).
- In a **main session** (direct chat with you) it auto-reads `SOUL.md` + `USER.md` + recent `memory/` + `MEMORY.md`. In shared/group contexts it does **not** load `MEMORY.md` (privacy).
- A command is just an instruction in chat: *"check my calendar for tomorrow,"* *"summarize today's git changes,"* etc. Per `AGENTS.md` it acts without asking for routine/safe work, and **asks first** before anything that leaves the machine.

> Exact gateway API endpoint/payload isn't documented here — inspect the control UI or the OpenClaw CLI
> for the precise request format before scripting against `:18789`. Don't assume; verify against the live tool.

---

## 3. HEARTBEAT.md — recurring background tasks
The agent receives a periodic **heartbeat poll**; on each, it reads `~/.openclaw/workspace/HEARTBEAT.md`
and acts on whatever's listed (else replies `HEARTBEAT_OK` and does nothing).

- **Empty file = no heartbeat work** (skips the API call — saves tokens).
- Add a **small checklist** of things to check periodically. Keep it short (token burn).
- Use heartbeat when checks **batch** (email + calendar + notifications in one turn) and timing can **drift**.

Example `HEARTBEAT.md`:
```markdown
# Periodic checks (rotate, 2-4x/day)
- Any urgent unread email? If so, summarize + notify.
- Calendar events in next 24-48h? Flag anything <2h out.
- git status on ~/Desktop/Bayesian-AI — uncommitted work piling up?
```
Track state in `workspace/memory/heartbeat-state.json` so it doesn't repeat checks.

---

## 4. Cron — scheduled / isolated tasks
Use **cron** (not heartbeat) when you need:
- **exact timing** ("9:00 AM sharp every Monday"),
- **isolation** from main-session history,
- a **different model / thinking level** for the task,
- **one-shot** reminders ("in 20 minutes…"),
- output delivered **directly to a channel** without main-session involvement.

> Set cron jobs through OpenClaw's chat/UI/CLI. The exact cron command syntax isn't captured here —
> ask the agent ("create a cron job that …") or check the CLI/UI; don't invent the syntax.

---

## 5. Skills — adding tools
Skills are *how* the agent gets new capabilities (TTS, cameras, web, etc.). Each skill ships a `SKILL.md`
that defines its tools. Per `openclaw.json`, `skills.allowBundled` is currently `[]` (no bundled skills enabled).
- To give the agent a new tool: add/enable a skill (then it appears in the agent's toolset).
- Keep **your environment-specifics** (device names, SSH hosts, voices) in `workspace/TOOLS.md`, NOT in the skill —
  skills are shared, your setup is yours.

---

## 6. Shaping behavior (config-as-command)
You also "command" the agent by editing its defining files:
- **`SOUL.md`** — personality/voice. **`IDENTITY.md`** — who it is. **`USER.md`** — facts about you it should know.
- **`AGENTS.md`** — operating rules (red lines, when to speak, memory discipline). Edit to change standing behavior.
- **`MEMORY.md`** — curate long-term memory (main session only; privacy-sensitive).

---

## 7. Safety red lines (from `AGENTS.md`)
- Don't exfiltrate private data, ever.
- Don't run destructive commands without asking; prefer `trash` over `rm`.
- Sending email/posts or anything that **leaves the machine** → ask first.
- In group chats it's a participant, not your proxy.

---

## 8. Quick start (this machine)
1. **Fix the model** (§ top): `ollama pull llama3.3` *or* edit `openclaw.json` → `gemma4:latest`.
2. Confirm Ollama is up: `curl http://127.0.0.1:11434/api/tags`.
3. Open the control UI / channel and chat a command.
4. For recurring jobs → add to `HEARTBEAT.md`; for exact-timed/isolated → cron.
5. Record what works (exact gateway API, cron syntax, enabled skills) **back into this doc** as you confirm it.

> Anything in this doc marked "verify against the live CLI/UI" is intentionally not guessed — fill in the exact
> syntax here once you've confirmed it from OpenClaw itself.
