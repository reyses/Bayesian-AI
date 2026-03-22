---
name: IS/IS-NOT audit method
description: Before building anything new, audit what IS built and what IS NOT built. Prevents rebuilding existing work and identifies true gaps.
type: feedback
---

Before designing or building any new feature, always run an IS/IS-NOT audit first.

**Why:** Multiple times we rebuilt things that already existed (lookback geometry, seed features, observer data) or planned features that depended on gaps we didn't know about (seed lookback prices never populated). The IS/IS-NOT method catches this upfront.

**How to apply:**
1. List every component the feature needs
2. For each: check if it EXISTS (with what data) or is MISSING (what's needed)
3. Identify the CRITICAL GAP — the one missing piece that blocks everything
4. Plan to fill the gap FIRST, then build on top

This applies to: new research tools, pipeline changes, observer integration, seed enrichment, any new module.
