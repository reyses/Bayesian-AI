import pandas as pd
df = pd.read_parquet('DATA/ATLAS/1s/2024_01_02.parquet')
print("Columns in 1s data:", df.columns.tolist())
print(df.head(2))
