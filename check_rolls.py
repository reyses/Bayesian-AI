import pandas as pd
df = pd.read_csv('DATA/ATLAS/roll_manifest.csv')
for d in ['2024_03_11', '2024_06_17', '2024_09_16', '2024_12_16']:
    r = df[df['day'] == d]
    if not r.empty:
        r = r.iloc[0]
        print(f'{d}: rolled={r["rolled"]}, fallback={r["calendar_fallback"]}, chosen={r["chosen"]}')
    else:
        print(f'{d}: NOT FOUND')
