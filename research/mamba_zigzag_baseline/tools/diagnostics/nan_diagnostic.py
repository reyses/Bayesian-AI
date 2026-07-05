import os
import sys
import numpy as np

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem
from core_v2.features import TF_HIERARCHY_V2, FEATURE_NAMES

def run_diagnostic():
    days = ["2025_11_14", "2025_11_17", "2025_11_18", "2025_11_19", "2025_11_20"]
    
    print(f"Running NaN Diagnostic Pass on Days: {days}")
    
    fps = MultiDayForwardPassSystem(
        atlas_root="DATA/ATLAS",
        features_root="DATA/ATLAS/FEATURES_5s_v2",
        labels_csv=None,
        days=days
    )
    
    iterator = iter(fps)
    
    nan_reports = []
    
    day_step_count = {}
    current_day = None
    
    try:
        while True:
            bar = next(iterator)
            
            # Keep track of session position
            if current_day != bar.day:
                current_day = bar.day
                day_step_count[current_day] = 0
            
            day_step_count[current_day] += 1
            
            if bar.v2_vector is not None:
                # Check for NaNs
                if np.isnan(bar.v2_vector).any():
                    # Find which index has NaNs
                    nan_indices = np.where(np.isnan(bar.v2_vector))[0]
                    # Map back to TF and feature
                    # TF_HIERARCHY_V2 has 9 TFs, each 4 channels
                    for idx in nan_indices:
                        feature_name = FEATURE_NAMES[idx]
                        
                        position = "LEADING_EDGE" if day_step_count[current_day] < 2000 else "MID_SESSION"
                        
                        nan_reports.append({
                            'day': current_day,
                            'bar_index': day_step_count[current_day],
                            'feature_name': feature_name,
                            'position': position
                        })
    except StopIteration:
        pass

    import pandas as pd
    if len(nan_reports) > 0:
        df = pd.DataFrame(nan_reports)
        print("\nNaN Diagnostic Table:")
        print(df.groupby(['day', 'feature_name', 'position']).size().reset_index(name='count'))
        df.to_csv('nan_diagnostic_report.csv', index=False)
        print("\nFull log saved to nan_diagnostic_report.csv")
    else:
        print("\nSUCCESS: No NaNs detected in the specified days.")

if __name__ == "__main__":
    run_diagnostic()
