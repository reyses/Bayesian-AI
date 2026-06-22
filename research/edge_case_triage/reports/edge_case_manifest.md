# Edge-case teaching set — 40 trades for 3-way verification → Gemma few-shot

Plots in reports/findings/edge_cases/. Columns verify_* are for Gemini/Claude/human to fill.

archetype counts (full 5k): STOPPED=2159, SMALL_LOSS=1270, SMALL_WIN=1064, CLEAN_RIDE=458, GAVE_BACK=454, CHOP=58

| tid | arch | dir | net$ | mfe | dur(m) | Claude: entry / exit | note |
|---|---|---|---|---|---|---|---|
| 3663 | CLEAN_RIDE | LONG | 294.0 | 163.8 | 367.0 | ok / ok | caught a real move, kept most of it |
| 378 | CLEAN_RIDE | LONG | 308.5 | 235.5 | 612.4 | ok / ok | caught a real move, kept most of it |
| 4041 | CLEAN_RIDE | SHORT | 486.0 | 325.0 | 351.4 | ok / ok | caught a real move, kept most of it |
| 2475 | CLEAN_RIDE | SHORT | 426.0 | 293.8 | 105.1 | ok / ok | caught a real move, kept most of it |
| 1742 | CLEAN_RIDE | LONG | 539.0 | 274.0 | 671.4 | ok / ok | caught a real move, kept most of it |
| 2252 | CLEAN_RIDE | LONG | 250.5 | 207.5 | 50.8 | ok / ok | caught a real move, kept most of it |
| 3529 | GAVE_BACK | SHORT | 45.0 | 117.2 | 71.3 | ok-but-late / too-wide (79pt trail) | real move, surrendered most of peak |
| 581 | GAVE_BACK | LONG | 76.0 | 118.8 | 681.0 | ok-but-late / too-wide (79pt trail) | real move, surrendered most of peak |
| 4174 | GAVE_BACK | SHORT | 66.0 | 114.5 | 86.5 | ok-but-late / too-wide (79pt trail) | real move, surrendered most of peak |
| 4912 | GAVE_BACK | LONG | 68.0 | 115.5 | 43.4 | ok-but-late / too-wide (79pt trail) | real move, surrendered most of peak |
| 2074 | GAVE_BACK | SHORT | 104.5 | 134.2 | 768.1 | ok-but-late / too-wide (79pt trail) | real move, surrendered most of peak |
| 3000 | GAVE_BACK | LONG | 75.5 | 118.5 | 289.0 | ok-but-late / too-wide (79pt trail) | real move, surrendered most of peak |
| 19 | CHOP | LONG | -10.5 | 4.2 | 59.7 | questionable / n/a | never developed (MFE<10pt) — likely a false-start entry |
| 235 | CHOP | SHORT | -84.0 | 3.8 | 87.5 | questionable / n/a | never developed (MFE<10pt) — likely a false-start entry |
| 3483 | CHOP | LONG | -40.5 | 6.5 | 57.2 | questionable / n/a | never developed (MFE<10pt) — likely a false-start entry |
| 1050 | CHOP | SHORT | -4.0 | 8.5 | 61.8 | questionable / n/a | never developed (MFE<10pt) — likely a false-start entry |
| 4227 | CHOP | SHORT | -33.0 | 4.5 | 21.1 | questionable / n/a | never developed (MFE<10pt) — likely a false-start entry |
| 5034 | CHOP | LONG | -39.5 | 9.5 | 53.1 | questionable / n/a | never developed (MFE<10pt) — likely a false-start entry |
| 5255 | STOPPED | SHORT | -103.5 | 12.0 | 58.2 | questionable / stop (mechanically ok) | reversed to -50pt stop; check if bad entry |
| 1743 | STOPPED | SHORT | -102.5 | 13.2 | 245.4 | questionable / stop (mechanically ok) | reversed to -50pt stop; check if bad entry |
| 5243 | STOPPED | SHORT | -111.5 | 3.0 | 0.6 | questionable / stop (mechanically ok) | reversed to -50pt stop; check if bad entry |
| 2848 | STOPPED | LONG | -103.5 | 35.5 | 1.0 | questionable / stop (mechanically ok) | reversed to -50pt stop; check if bad entry |
| 2028 | STOPPED | LONG | -103.0 | 0.0 | 3.3 | questionable / stop (mechanically ok) | reversed to -50pt stop; check if bad entry |
| 4505 | STOPPED | LONG | -106.0 | 6.8 | 18.6 | questionable / stop (mechanically ok) | reversed to -50pt stop; check if bad entry |
| 118 | SMALL_WIN | LONG | 32.5 | 27.8 | 238.1 | ok? / ok? | marginal |
| 5404 | SMALL_WIN | LONG | 18.5 | 91.0 | 50.0 | ok? / ok? | marginal |
| 1963 | SMALL_WIN | LONG | 208.0 | 185.0 | 45.5 | ok? / ok? | marginal |
| 3738 | SMALL_WIN | SHORT | 131.0 | 147.0 | 118.3 | ok? / ok? | marginal |
| 3984 | SMALL_WIN | SHORT | 9.0 | 86.8 | 10.5 | ok? / ok? | marginal |
| 2795 | SMALL_WIN | LONG | 134.0 | 148.5 | 6.0 | ok? / ok? | marginal |
| 5257 | SMALL_LOSS | SHORT | -71.5 | 47.5 | 50.5 | ok? / ok? | marginal |
| 5020 | SMALL_LOSS | LONG | -19.5 | 71.0 | 4.5 | ok? / ok? | marginal |
| 1860 | SMALL_LOSS | SHORT | -47.5 | 58.0 | 21.5 | ok? / ok? | marginal |
| 4270 | SMALL_LOSS | SHORT | -56.0 | 53.0 | 38.9 | ok? / ok? | marginal |
| 3449 | SMALL_LOSS | SHORT | -89.5 | 36.2 | 13.2 | ok? / ok? | marginal |
| 3423 | SMALL_LOSS | LONG | -1.0 | 82.8 | 179.6 | ok? / ok? | marginal |
| 1938 | CLEAN_RIDE | SHORT | 1557.5 | 859.8 | 612.5 | ok / ok | caught a real move, kept most of it |
| 3041 | STOPPED | LONG | -454.0 | 33.8 | 189.5 | questionable / stop (mechanically ok) | reversed to -50pt stop; check if bad entry |
| 4101 | CLEAN_RIDE | LONG | 523.5 | 285.2 | 1387.3 | ok / ok | caught a real move, kept most of it |
| 1257 | STOPPED | SHORT | -253.5 | 0.0 | 0.0 | questionable / stop (mechanically ok) | reversed to -50pt stop; check if bad entry |