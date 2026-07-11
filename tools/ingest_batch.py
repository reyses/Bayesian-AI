import sys
import json
import os

if len(sys.argv) < 2:
    print("Usage: python ingest_batch.py <batch_num>")
    sys.exit(1)

batch_num = int(sys.argv[1])
batch_size = 10

# Load remaining files
try:
    with open('research/nt8_catalog/tools/ingest_state/remaining_tech.json', 'r') as f:
        files = json.load(f)
except FileNotFoundError:
    print("remaining_tech.json not found.")
    sys.exit(1)

# Calculate indices
start_idx = (batch_num - 2) * batch_size # batch 2 starts at index 0 of remaining_tech
end_idx = start_idx + batch_size

if start_idx >= len(files):
    print("ALL_DONE")
    sys.exit(0)

batch_files = files[start_idx:end_idx]
out_path = 'research/nt8_catalog/tools/ingest_state/current_batch.txt'

keywords = ['setup', 'entry', 'exit', 'stop', 'target', 'indicator', 'volume', 'macd', 'rsi', 'vwap', 'z-score', 'trend', 'support', 'resistance', 'pattern', 'strategy', 'signal', 'reversal']

with open(out_path, 'w', encoding='utf-8') as out:
    for filename in batch_files:
        filepath = os.path.join('research/nt8_catalog', filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        paragraphs = text.split('\n\n')
        signals = []
        for p in paragraphs:
            p = p.strip()
            if len(p) < 30 or p.startswith('* ['):
                continue
            
            p_lower = p.lower()
            if any(k in p_lower for k in keywords):
                # Clean up markdown image links to save tokens
                import re
                p = re.sub(r'!\[.*?\]\(.*?\)', '', p)
                if len(p.strip()) > 20:
                    signals.append(p.strip())
        
        if signals:
            out.write(f'--- FILE: {filename} ---\n')
            out.write('\n\n'.join(signals))
            out.write('\n\n')

print(f"Extracted {len(batch_files)} files into current_batch.txt")
