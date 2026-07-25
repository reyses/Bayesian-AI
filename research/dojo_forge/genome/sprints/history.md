sprint1 — FAIL (selectivity 100% vs bar <=80%; info_rate 100% passed)
diagnosis: model memo'd every single frame (18/18), ignoring "MOST frames deserve NO memo," and the memos were near-verbatim reruns of the overlay's own format exemplar (reversion_prob/multi-TF-alignment template) rather than genuinely novel signatures.
change: added a hard <1-in-5 memo cap with zero-score threat, an explicit BAD exemplar matching the observed spam pattern plus a GOOD exemplar with a decision-relevant pattern (not a decision restatement), and a self-count-before-writing instruction.

sprint2 — FAIL (selectivity 100% vs bar <=80%; info_rate 100% passed)
diagnosis: the sprint1 cap failed because "silently count memos so far, skip if near 1-in-5" asks for state the model can't reliably track across frames — instead the model wrote a data-bearing memo every frame because each frame's literal reversion_prob/velocity numbers looked "new," even though all 12 memos restated the same underlying class (high reversion_prob + weak velocity flip = noise); it confused new numbers for a new pattern.
change: replaced counting-based selectivity with a class-dedup rule (default MEMO: NONE; a memo is a new pattern-CLASS, not new numbers — same class recurring with different values still gets MEMO: NONE) plus a zero-score exemplar built from this sprint's exact failure and an explicit "is this class already logged?" self-check before writing.
SESSION STAGE (post-sprint3, 2026-07-24 ~22:3x) — sprint3 REGRESSED (info 100%->0%:
retro's anti-spam exemplar broke content). Diagnosis from data, no interview needed:
capability proven (sprint1), instruction-side selectivity unreachable at temp-0
(pattern completion), retro pressure destructive. FIX: restored sprint-1 overlay
(proven content) + GUARD C curation cap=4 + near-dup rejection (selectivity by
construction, bank-basis scoring, documented redesign). Relaunching as sprint 4.
