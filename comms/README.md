# Claude ⇄ Gemini file mailbox — protocol

A low-tech, no-API, no-cost communication channel between Claude (Claude Code) and Gemini
(the user's flat-rate Gemini interface). Both agents read/write one append-only file,
`comms/mailbox.md`. This avoids the metered Gemini API **and** the MCP auto-exfiltration risk:
nothing is sent anywhere automatically — Gemini only sees what is *deliberately written into
the mailbox*. The user is the supervisor of the channel.

## Why it exists
- **Spare the Claude (Anthropic) session limit**: offload INPUT-heavy / OUTPUT-light work to
  Gemini's quota — Gemini reads the big files and writes back a tight result.
- **Spare the Gemini API cost**: Gemini runs on the user's flat-rate interface, polling the
  file, not the metered API.

## Message format (in `comms/mailbox.md`, append-only)
Each message is one block. Reply by appending a NEW block with the same `ID`, `FROM` swapped,
and `STATUS: done`. Never edit prior blocks (append-only = full auditable history).

```
## MSG <id> | FROM: claude | TO: gemini | STATUS: pending | <YYYY-MM-DD HH:MM>
TASK: <explicit instruction; ask for a tight, structured result>
FILES: <comma-separated repo-relative paths Gemini should READ on its quota; optional>
RETURN: <exactly what to put in the reply, e.g. "a 10-line summary" / "a table" / "PASS|FAIL + 3 reasons">
---
## MSG <id> | FROM: gemini | TO: claude | STATUS: done | <YYYY-MM-DD HH:MM>
RESPONSE:
<Gemini's tight result here>
---
```

## Who does what
- **Claude (request side):** append a `FROM: claude TO: gemini STATUS: pending` block, then
  END THE TURN. **Do NOT run a tight Claude poll-loop** — each loop tick burns Claude tokens and
  defeats the purpose. Get poked when Gemini is done (see "notification") or schedule ONE delayed
  wakeup to check back.
- **Gemini (worker side):** a recurring task in the user's Gemini tool watches `mailbox.md`,
  finds the newest `TO: gemini STATUS: pending` block with no matching reply, READS the listed
  FILES (on Google's quota), and appends a `FROM: gemini STATUS: done` reply with the same ID.
- **User (supervisor):** owns what enters the mailbox; can read the whole conversation; pokes
  Claude when a reply lands (or via the existing Telegram inject channel).

## Rules / guardrails
1. **Gemini gets verifiable, bounded jobs only** — bulk read / summarize / extract / parse
   run-output / first-pass audit. NOT rigor-critical decisions (firewall / CI / no-lookahead).
   Claude VERIFIES every Gemini result (trust-but-check, like a subagent).
2. **One-way delegation** (Claude asks → Gemini answers). No autonomous back-and-forth chatter.
3. **Curation = the exfiltration control.** Anything written under FILES/TASK is what Gemini
   (Google) sees. Do not put secrets or whole proprietary modules in unless intended. The
   automated "ship any file" path is deliberately absent here.
4. **STATUS + ID** prevent double-processing and correlate request↔reply.
5. **Worth it for BIG, infrequent offloads** (re-read the 262KB corpus and extract X; scan all
   VM logs; bulk-audit a folder). Not worth the coordination overhead for small/frequent tasks
   — use a normal Claude subagent for those.
