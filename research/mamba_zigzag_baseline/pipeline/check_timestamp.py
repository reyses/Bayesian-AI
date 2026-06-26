import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from mamba_env import MambaRLTradingEnv
import datetime

atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
features_root = os.path.join(atlas_root, "FEATURES_5s_v2")
labels_csv = os.path.join(atlas_root, "regime_labels_2d.csv")
days = ["2024_02_20"]

env = MambaRLTradingEnv(atlas_root, features_root, labels_csv, days)
env.reset()
print(f"Timestamp type: {type(env.current_bar.timestamp)}")
print(f"Timestamp value: {env.current_bar.timestamp}")
# convert to datetime
if isinstance(env.current_bar.timestamp, (int, float, np.integer, np.floating)):
    dt = datetime.datetime.fromtimestamp(env.current_bar.timestamp / 1e9 if env.current_bar.timestamp > 1e12 else env.current_bar.timestamp, tz=datetime.timezone.utc)
    print(f"Datetime: {dt}")
elif hasattr(env.current_bar.timestamp, 'to_pydatetime'):
    print(f"Datetime: {env.current_bar.timestamp.to_pydatetime()}")
