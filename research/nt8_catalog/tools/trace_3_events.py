import os
import glob
import pandas as pd

def trace():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    of_events_path = os.path.join(base_dir, 'tests/ORDERFLOW-14/events.parquet')
    
    if not os.path.exists(of_events_path):
        print(f"Waiting for {of_events_path} to be generated...")
        return
        
    df_events = pd.read_parquet(of_events_path)
    
    # Take 3 random events
    if len(df_events) < 3:
        samples = df_events
    else:
        samples = df_events.sample(3, random_state=42)
        
    l0_dir = os.path.abspath(os.path.join(base_dir, '../../DATA/ATLAS/5s'))
    
    out_lines = []
    out_lines.append("# Event Trace Verification")
    out_lines.append("Verifying `resolution_idx` relative to `event_idx` for 3 sampled events from ORDERFLOW-14.")
    out_lines.append("")
    
    for _, row in samples.iterrows():
        day = row['day']
        event_idx = int(row['event_idx'])
        resolution_idx = int(row['resolution_idx'])
        dur = int(row['duration_bars'])
        
        pq_path = os.path.join(l0_dir, f"{day.replace('-', '_')}.parquet")
        if not os.path.exists(pq_path):
            print(f"File not found: {pq_path}")
            continue
        print(f"Found {pq_path}")
            
        df_day = pd.read_parquet(pq_path)
        df_day['dt'] = pd.to_datetime(df_day['timestamp'])
        df_day = df_day.sort_values('dt').reset_index(drop=True)
        
        entry_bar = df_day.iloc[event_idx]
        exit_bar = df_day.iloc[resolution_idx] if resolution_idx != -1 else None
        
        out_lines.append(f"## Event on {day} (Setup {row['setup']})")
        out_lines.append(f"- **Entry Index**: {event_idx} (dt: {entry_bar['dt']})")
        if exit_bar is not None:
            out_lines.append(f"- **Exit Index**: {resolution_idx} (dt: {exit_bar['dt']})")
            out_lines.append(f"- **Calculated Duration**: {dur} bars (Difference: {resolution_idx - event_idx} bars)")
        else:
            out_lines.append(f"- **Exit Index**: {resolution_idx} (Time Expired / No Hit)")
        out_lines.append(f"- **Recorded Depth**: {row['depth']:.4f}")
        out_lines.append(f"- **Recorded Magnitude**: {row['magnitude']:.2f}")
        out_lines.append("")
        
    out_path = os.path.join(base_dir, 'reports/AG_cat_00_EVENT_TRACE.md')
    with open(out_path, 'w') as f:
        f.write("\n".join(out_lines))
        
    print(f"Trace written to {out_path}")

if __name__ == '__main__':
    trace()
