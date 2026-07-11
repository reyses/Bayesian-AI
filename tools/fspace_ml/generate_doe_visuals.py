import os
import pandas as pd
import numpy as np
import glob
import re
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
catalog_dir = os.path.join(base_dir, 'research', 'nt8_catalog', 'tests')
checkpoints_dir = os.path.join(base_dir, 'DATA', 'ATLAS', 'ML_CHECKPOINTS')

def process_strategy(s_dir):
    s_name = os.path.basename(s_dir)
    report_path = os.path.join(s_dir, 'augmentation', 'fspace_doe_report.md')
    if not os.path.exists(report_path):
        return
        
    with open(report_path, 'r') as f:
        content = f.read()
        
    # Extract selected features
    match = re.search(r'### Top Selected Features \(Stepwise Forward Elimination\)\n(.*)', content, re.DOTALL)
    if not match: return
    
    features = []
    for line in match.group(1).split('\n'):
        if line.startswith('- `'):
            feat = line.replace('- `', '').replace('`', '').strip()
            if feat: features.append(feat)
            
    if not features: return
    
    # Load data
    parquet_files = glob.glob(os.path.join(checkpoints_dir, f"{s_name}_*.parquet"))
    if not parquet_files: return
    
    dfs = []
    for p in parquet_files:
        if os.path.getsize(p) > 2000:
            dfs.append(pd.read_parquet(p))
    
    if not dfs: return
    df = pd.concat(dfs, ignore_index=True)
    
    # Check if features exist
    missing = [f for f in features if f not in df.columns]
    features = [f for f in features if f in df.columns]
    
    if not features: return
    
    df = df.fillna(0) # Scrub NaNs which PyTorch ignored but statsmodels crashes on
    
    X = df[features].copy()
    target_col = 'hit' if 'hit' in df.columns else 'Hit'
    y = df[target_col].copy()
    
    # Standardize X to get standardized effects
    X_std = (X - X.mean()) / X.std()
    X_std = sm.add_constant(X_std)
    
    try:
        model = sm.Logit(y, X_std)
        result = model.fit(disp=0)
    except Exception as e:
        print(f"Failed to fit Logit for {s_name}: {e}")
        return
        
    # Get stats
    aic = result.aic
    bic = result.bic
    llr_pvalue = result.llr_pvalue
    
    # Pseudo R2 (McFadden)
    pseudo_r2 = result.prsquared
    
    # Base win rate
    base_wr = y.mean() * 100
    
    # Augmented win rate (top decile of predictions)
    preds = result.predict(X_std)
    df['pred'] = preds
    top_decile = df[df['pred'] > df['pred'].quantile(0.9)]
    aug_wr = top_decile[target_col].mean() * 100
    
    # Pareto Chart
    params = result.params.drop('const')
    pvalues = result.pvalues.drop('const')
    
    effects = pd.DataFrame({
        'Feature': params.index,
        'Effect': params.values,
        'AbsEffect': np.abs(params.values),
        'PValue': pvalues.values
    })
    effects = effects.sort_values('AbsEffect', ascending=True)
    
    plt.figure(figsize=(10, 8))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in effects['Effect']]
    plt.barh(effects['Feature'], effects['AbsEffect'], color=colors, alpha=0.8)
    plt.axvline(0, color='black', lw=1)
    plt.title(f"Standardized Effects Pareto (Log-Odds) - {s_name}")
    plt.xlabel("Absolute Standardized Coefficient")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add p-value markers
    for i, pval in enumerate(effects['PValue']):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        plt.text(effects['AbsEffect'].iloc[i] + 0.01, i, sig, va='center', fontweight='bold', color='white')
        
    plt.tight_layout()
    pareto_path = os.path.join(s_dir, 'augmentation', 'pareto_effects.png')
    plt.savefig(pareto_path, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    
    # Interaction Plot (Top 2 Features)
    top2 = effects.sort_values('AbsEffect', ascending=False)['Feature'].head(2).tolist()
    interaction_path = None
    if len(top2) >= 2:
        f1, f2 = top2[0], top2[1]
        plt.figure(figsize=(8, 6))
        
        sns.kdeplot(data=df, x=f1, y=f2, hue=target_col, fill=True, alpha=0.5, palette=['#e74c3c', '#2ecc71'])
        plt.title(f"Interaction: {f1}\nvs\n{f2}")
        plt.tight_layout()
        interaction_path = os.path.join(s_dir, 'augmentation', 'interaction_plot.png')
        plt.savefig(interaction_path, dpi=150, facecolor='#111111', edgecolor='none')
        plt.close()

    # Update Report
    content = re.sub(r'\* \*\*AIC:\*\* \[Pending\]', f'* **AIC:** {aic:.2f}', content)
    content = re.sub(r'\* \*\*BIC:\*\* \[Pending\]', f'* **BIC:** {bic:.2f}', content)
    content = re.sub(r'\* \*\*Pseudo R-Squared:\*\* \[Pending\]', f'* **Pseudo R-Squared:** {pseudo_r2:.4f}', content)
    content = re.sub(r'\* \*\*Baseline Win Rate:\*\* \[Pending\]', f'* **Baseline Win Rate:** {base_wr:.2f}%', content)
    content = re.sub(r'\* \*\*Augmented Predictive Win Rate:\*\* \[Pending\]', f'* **Augmented Predictive Win Rate:** {aug_wr:.2f}% (Top 10% Decile Threshold)', content)
    
    # Also update the Visuals pending
    pareto_link = f"![Standardized Effects Pareto](./pareto_effects.png)"
    content = re.sub(r'\*\(Visuals pending script generation\)\*\n\* \*\*Top Linear Effects:\*\* \[Pending\]\n\* \*\*Top Quadratic Effects:\*\* \[Pending\]\n\* \*\*Top Cubic Effects:\*\* \[Pending\]', pareto_link, content)
    
    if interaction_path:
        int_link = f"![Interaction Plot Top 2 Features](./interaction_plot.png)"
        content = re.sub(r'\*\(Visuals pending script generation\)\*\n\* \*\*Interaction 1:\*\* \[Pending\]\n\* \*\*Interaction 2:\*\* \[Pending\]', int_link, content)
        
    with open(report_path, 'w') as f:
        f.write(content)
        
    print(f"Processed {s_name}. Aug WR: {aug_wr:.2f}%. AIC: {aic:.2f}")

for folder in os.listdir(catalog_dir):
    s_dir = os.path.join(catalog_dir, folder)
    if os.path.isdir(s_dir):
        process_strategy(s_dir)
