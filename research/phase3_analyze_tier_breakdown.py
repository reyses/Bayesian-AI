import json

def analyze_buckets_by_tier():
    with open('artifacts/regime_buckets.json', 'r') as f:
        buckets = json.load(f)
        
    print("\n--- TIER BREAKDOWN FOR TOP 5 REGIMES ---")
    for i in range(1, 6):
        b = buckets[str(i)]
        print(f"Regime {i}: Root Segment #{b['root_segment']}")
        print(f"  Total Members: {b['total_members']}")
        print(f"  Tier 1 (Flawless): {len(b['members_tier_1'])}")
        print(f"  Tier 2 (Great):    {len(b['members_tier_2'])}")
        print(f"  Tier 3 (Good):     {len(b['members_tier_3'])}")
        print(f"  Tier 4 (Edge):     {len(b['members_tier_4'])}\n")
        
if __name__ == '__main__':
    analyze_buckets_by_tier()
