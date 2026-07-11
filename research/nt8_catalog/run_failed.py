import os
import subprocess

failed = [
    r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests\ATR-09_Statistical_Fade\ag_deepdive_09_atr.py",
    r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests\DOW-19_Price_Volume_Divergence\ag_deepdive_19_dow.py",
    r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests\FIB-17_Confluence\ag_deepdive_17_fib.py",
    r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests\HNS-22_Head_And_Shoulders_Volume\ag_deepdive_22_hns.py",
    r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests\ORDERFLOW-14\ag_deepdive_14_orderflow.py",
    r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests\VP-01_Volume_Profile\ag_deepdive_01_vol_profile.py"
]

for f in failed:
    print(f"Running {os.path.basename(f)}...")
    subprocess.run(["python", f], cwd=os.path.dirname(f), check=True)
    print("Done.\n")
