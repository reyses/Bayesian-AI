import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

log_files = glob.glob(r"C:\Users\reyse\.gemini\antigravity\brain\0b405af3-d525-4c87-b71d-cb77ea225a55\.system_generated\tasks\*.log")

results = []
pattern = re.compile(r"Completed (\d{4}_\d{2}_\d{2})\. Avg Loss: ([\d\.]+)")

for log_file in sorted(log_files):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                date = match.group(1)
                loss = float(match.group(2))
                results.append({"date": date, "loss": loss})

df = pd.DataFrame(results)

if len(df) == 0:
    print("No loss data found!")
    exit(1)

# In case of duplicate runs over the same date due to restarts, take the last one
df = df.drop_duplicates(subset=['date'], keep='last')
df = df.sort_values("date").reset_index(drop=True)

csv_path = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\mamba_zigzag_baseline\reports\loss_curve.csv"
df.to_csv(csv_path, index=False)
print(f"Extracted {len(df)} loss data points to {csv_path}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["loss"], label="CrossEntropyLoss", color="blue", alpha=0.7)
plt.title("Mamba Dynamic ZigZag Baseline - Training Loss")
plt.xlabel("Trading Day (Index)")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = r"C:\Users\reyse\.gemini\antigravity\brain\0b405af3-d525-4c87-b71d-cb77ea225a55\mamba_loss_curve.png"
plt.savefig(plot_path)
print(f"Saved plot to {plot_path}")
