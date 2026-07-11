import os
import pandas as pd

base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\DATA\ATLAS"
parquet_path = os.path.join(base_dir, "order_flow_delta_5s.parquet")

df = pd.read_parquet(parquet_path)
df = df[df['close'] > 10000].copy()
if isinstance(df.index, pd.DatetimeIndex):
    df['dt'] = df.index.tz_convert('America/Chicago')
else:
    df['dt'] = pd.to_datetime(df.index, utc=True).tz_convert('America/Chicago')
df['day_str'] = df['dt'].dt.strftime('%Y-%m-%d')
df_day = df[df['day_str'] == '2025-07-30'].copy()
df_rth = df_day[(df_day['dt'].dt.time >= pd.Timestamp('08:30').time()) & (df_day['dt'].dt.time <= pd.Timestamp('15:15').time())].copy()


# Print the 5 bars around idx 2765 of the RTH data
# Wait, was the idx 2765 relative to the RTH block or the whole day?
# In the scripts, prices = df_day['close'].values, so it's relative to RTH block.

start = 2765
end = 2765 + 61
subset = df_rth.iloc[start : end]
print("--- PRICE PATH FROM IDX 2765 TO 2826 ---")
print(subset[['dt', 'open', 'high', 'low', 'close', 'volume']])
print(f"Max close: {subset['close'].max()}, Min close: {subset['close'].min()}")

