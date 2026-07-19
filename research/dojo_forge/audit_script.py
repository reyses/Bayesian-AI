import json
import codecs

with open('C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/dojo_forge/reports/truncation_audit_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

total = len(data)
contaminated = sum(1 for d in data if d['tainted'] == 'Y')

table_rows = ["| Episode ID | True Prompt Tokens | Effective Ctx | Tainted? | Re-run? |", "|---|---|---|---|---|"]
for d in data:
    table_rows.append(f"| {d['eid']} | {d['true_tokens']} | {d['effective_ctx']} | {d['tainted']} | {d['rerun']} |")

table_str = '\n'.join(table_rows)

inventory = '''
## (c) Gen-0 Inventory
- **Lane**: gemma-fallback (gemma4:e2b)
- **Episodes Played**: 145
- **Status**: Designated as CONTROL-ARM DATASET (over-fills the 20-episode control).
- **Primary Lane**: qwen3:14b native (Currently running on CPU pathway, 156 packets in total).
- **Exclusion Set Note**: 145 episodes were played from the available day pool (disjoint from dev-rotation holdout and terminal lockbox as mandated by the Ride-Edge Gate Spec).
'''

md_content = f'''# AG AUDIT REFILE - Context Truncation (Doc 118)
**Doc:** 118 * **Date:** 2026-07-19 * **Author:** AG * **For:** Claude Fable (reviewer)

## (a) Actual Contamination Count
- **Total Episodes Evaluated in Fallback Gen-0:** {total}
- **Contaminated Episodes (Prompt > 4096):** {contaminated}
- **Contamination Rate:** {(contaminated/total)*100:.1f}%

## (b) Per-Episode Audit Table
{table_str}
{inventory}
'''

with codecs.open('C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/nt8_catalog/comms/118_2026-07-19_AG_AUDIT_REFILE.md', 'w', encoding='utf-8') as f:
    f.write(md_content)
