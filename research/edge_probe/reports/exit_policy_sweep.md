# Causal exit-policy sweep (full ATLAS, dossier entries)
6,139 RIDE trades, 520 days. Same entries; exit swapped. Never-bail on losers kept (no stop).

| policy | trades | net $ | $/day | 95% CI | sig | PF |
|---|---|---|---|---|---|---|
| TP30 | 10,308 | $23,856 | $45.9 | [$12.4, $78.8] | **YES** | 1.071 |
| TP40 | 9,133 | $18,519 | $35.6 | [$1.7, $69.2] | **YES** | 1.059 |
| TP20 | 12,253 | $17,858 | $34.3 | [$5.6, $64.8] | **YES** | 1.050 |
| ARM30TR6 | 9,170 | $16,734 | $32.2 | [$-1.5, $65.5] | no | 1.057 |
| TP50 | 8,366 | $16,130 | $31.0 | [$-2.4, $64.3] | no | 1.054 |
| ARM40TR15 | 7,876 | $15,125 | $29.1 | [$-2.3, $62.9] | no | 1.057 |
| RIDE | 6,139 | $15,006 | $28.9 | [$-2.2, $61.2] | no | 1.071 |
| ARM30TR15 | 8,341 | $13,202 | $25.4 | [$-6.6, $57.0] | no | 1.049 |
| ARM30TR10 | 8,815 | $11,870 | $22.8 | [$-8.0, $54.2] | no | 1.042 |
| ARM40TR6 | 8,413 | $11,192 | $21.5 | [$-10.1, $55.3] | no | 1.039 |
| ARM40TR10 | 8,185 | $9,186 | $17.7 | [$-14.2, $50.2] | no | 1.033 |

RIDE = current R-trigger never-bail baseline (~$16k / +$31/day, not sig). TP/ARM harvest the ~+43pt peak causally. A policy that beats RIDE with a CI excluding 0 is a real exit edge; if none do, the peak is not causally capturable (bar@MFE varies too much to arm on).
