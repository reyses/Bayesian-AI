import json

def analyze_buckets():
    print("[Analysis] Loading regime buckets...")
    with open('artifacts/regime_buckets.json', 'r') as f:
        buckets = json.load(f)
        
    print(f"Total Buckets: {len(buckets)}")
    
    # Buckets are naturally ordered by their ID, but let's sort by total_members just to be safe
    # Actually, they were extracted based on tier 1/2 degree, so Bucket 1 is the most "dominant" root.
    
    print("\n--- TOP 15 DOMINANT MARKET REGIMES ---")
    for i in range(1, 16):
        b_id = str(i)
        if b_id in buckets:
            b = buckets[b_id]
            print(f"Regime {b_id:>2}: Root Segment #{b['root_segment']:<5} | Encompasses {b['total_members']:>5} unique MNQ segments")
            
    print("\n--- DISTRIBUTION ---")
    sizes = [b['total_members'] for b in buckets.values()]
    print(f"Top 1% of Regimes (Top 30) encompass {sum(sizes[:30])} segments.")
    print(f"Top 500 Regimes encompass {sum(sizes[:500])} segments.")
    print(f"The remaining 2529 Regimes encompass only {sum(sizes[500:])} segments.")
    
if __name__ == '__main__':
    analyze_buckets()
