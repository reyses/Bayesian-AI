import pandas as pd
df = pd.read_parquet('c:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/nt8_catalog/tests/OHLC-01_Prior_Day/events.parquet')
print("Total rows:", len(df))
print(df[['event_idx', 'resolution_idx', 'duration_bars', 'setup', 'hit']].head(20))
