import csv, sys
from collections import defaultdict, Counter

def money(s):
    s = s.strip().replace('$', '').replace(',', '')
    neg = s.startswith('(') and s.endswith(')')
    s = s.strip('()')
    v = float(s) if s else 0.0
    return -v if neg else v

path = sys.argv[1]
rows = [r for r in csv.DictReader(open(path, encoding='utf-8-sig')) if r.get('Trade number', '').strip()]
stops = [money(r['Profit']) for r in rows if r['Exit name'] in ('Stop loss', 'X_CatastrophicStop')]
print("STOP trades: %d  avg $%.0f  worst $%.0f  best $%.0f  (MNQ 50pt=$100, 100pt=$200)"
      % (len(stops), sum(stops)/len(stops), min(stops), max(stops)))
# distribution of stop losses in $100 buckets
buckets = Counter(int(abs(s)//50)*50 for s in stops)
print("stop-loss size buckets ($):", dict(sorted(buckets.items())))

day = defaultdict(float)
for r in rows:
    day[r['Entry time'].split(' ')[0]] += money(r['Profit'])
tot = sum(day.values())
srt = sorted(day.values(), reverse=True)
med = sorted(day.values())[len(day)//2]
print("net $%.0f | top1 $%.0f (%.0f%%) | top3 $%.0f (%.0f%%) | median-day $%.0f | net-ex-top1 $%.0f"
      % (tot, srt[0], 100*srt[0]/tot, sum(srt[:3]), 100*sum(srt[:3])/tot, med, tot-srt[0]))
fe = {}
for r in rows:
    d, t = r['Entry time'].split(' ', 1)
    fe.setdefault(d, t)
print("first-entry time mode:", Counter(fe.values()).most_common(3))
