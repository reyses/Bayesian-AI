import pandas as pd
import os

parquet_path = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\DATA\ATLAS\order_flow_delta_5s.parquet"
df = pd.read_parquet(parquet_path)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df[df['close'] > 10000].copy()
if isinstance(df.index, pd.DatetimeIndex):
    df['dt'] = df.index.tz_convert('America/Chicago')
else:
    df['dt'] = pd.to_datetime(df.index, utc=True).tz_convert('America/Chicago')
df['day_str'] = df['dt'].dt.strftime('%Y-%m-%d')
df_day = df[df['day_str'] == '2025-07-30'].sort_values('dt').copy()
df_rth = df_day[(df_day['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df_day['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()
df_rth = df_rth.reset_index(drop=True)

idx = 2765
print(f"RTH Length: {len(df_rth)}")
if idx < len(df_rth):
    path = df_rth.iloc[idx+1:idx+61]['close'].values
    print(f"p0 (idx {idx}): Close {df_rth.iloc[idx]['close']} at {df_rth.iloc[idx]['dt']}")
    print(f"Path Min: {path.min()}, Path Max: {path.max()}")
    print(path)
else:
    print(f"Index {idx} out of bounds")
