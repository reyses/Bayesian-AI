import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem

def create_quadratic_weights(window_size):
    w = np.array([(i / window_size)**2 for i in range(1, window_size + 1)])
    w -= np.mean(w)
    return w / np.sum(np.abs(w))

def main():
    ATLAS = 'DATA/ATLAS'
    # USING THE NEW 1-SECOND FEATURE SET
    FEAT = f'{ATLAS}/FEATURES_1s_v2'
    LABELS = f'{ATLAS}/regime_labels_2d.csv'

    fps = MultiDayForwardPassSystem(
        atlas_root=ATLAS,
        features_root=FEAT,
        labels_csv=LABELS,
        days=None  # Full dataset
    )
    
    # EXACT ORIGINAL 1-SECOND TUNING (4.8 minutes and 1.5 minutes)
    blue_w_arr = create_quadratic_weights(288)
    orange_w_arr = create_quadratic_weights(90)
    alpha = 2 / (3 + 1)
    
    history_prices = []
    last_p = None
    ema_v = 0.0
    
    pos = 0
    eprice = 0.0
    ets = 0
    mfe_px = 0.0
    mae_px = 0.0
    trades = []
    
    current_day = None
    ny_tz = pytz.timezone('America/New_York')
    
    print("Beginning TRUE 1s Forward Pass Stepping...")
    for state in fps:
        if state.day != current_day:
            print(f"Forward Pass Stepping: {state.day}")
            current_day = state.day
            history_prices.clear()
            last_p = None
            ema_v = 0.0
            pos = 0
        
        px = state.price
        ts = state.timestamp
        history_prices.append(px)
        if len(history_prices) > 300:
            history_prices.pop(0)
            
        # Track MFE and MAE
        if pos == 1:
            mfe_px = max(mfe_px, px)
            mae_px = min(mae_px, px)
        elif pos == -1:
            mfe_px = min(mfe_px, px)
            mae_px = max(mae_px, px)
            
        # End of Day Maintenance Gap Check
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ny_tz)
        if dt.hour == 16 and dt.minute >= 50:
            if pos != 0:
                pnl_open = (px - eprice) * 20 if pos == 1 else (eprice - px) * 20
                usd_net = pnl_open - 2.5
                mfe_usd = (mfe_px - eprice) * 20 if pos == 1 else (eprice - mfe_px) * 20
                mae_usd = (mae_px - eprice) * 20 if pos == 1 else (eprice - mae_px) * 20
                trades.append((current_day, ets, 'LONG' if pos==1 else 'SHORT', eprice, ts, px, pnl_open, usd_net, mfe_usd, mae_usd))
                pos = 0
            continue
        
        if last_p is not None:
            v_raw = px - last_p
            if len(history_prices) == 2:
                ema_v = v_raw
            else:
                ema_v = alpha * v_raw + (1 - alpha) * ema_v
        last_p = px
        
        if len(history_prices) < 288:
            continue
            
        prices_arr = np.array(history_prices)
        blue_slope = np.dot(blue_w_arr, prices_arr[-288:])
        orange_slope = np.dot(orange_w_arr, prices_arr[-90:])
        p_v = ema_v
        
        # Original pure vector logic
        if pos == 0:
            if orange_slope > 0 and blue_slope > 0 and p_v > 0:
                pos = 1
                eprice = px
                ets = ts
                mfe_px = px
                mae_px = px
            elif orange_slope < 0 and blue_slope < 0 and p_v < 0:
                pos = -1
                eprice = px
                ets = ts
                mfe_px = px
                mae_px = px
        else:
            pnl_open = (px - eprice) * 20 if pos == 1 else (eprice - px) * 20
            
            # Pure exit logic: Hard stop or Vector flip (NO aggressive TP to cause loops)
            hit_stop = pnl_open <= -100.0
            vector_flip = (pos == 1 and (orange_slope < 0 or p_v < 0)) or (pos == -1 and (orange_slope > 0 or p_v > 0))
            
            if hit_stop or vector_flip:
                usd_gross = pnl_open
                usd_net = usd_gross - 2.5
                mfe_usd = (mfe_px - eprice) * 20 if pos == 1 else (eprice - mfe_px) * 20
                mae_usd = (mae_px - eprice) * 20 if pos == 1 else (eprice - mae_px) * 20
                trades.append((state.day, ets, 'LONG' if pos==1 else 'SHORT', eprice, ts, px, usd_gross, usd_net, mfe_usd, mae_usd))
                pos = 0

    cols = ['day', 'entry_ts', 'leg_dir', 'entry_price', 'exit_ts', 'exit_price', 'gross_usd', 'net_usd', 'mfe_usd', 'mae_usd']
    df = pd.DataFrame(trades, columns=cols)
    
    os.makedirs('reports/findings', exist_ok=True)
    df.to_csv('reports/findings/nmp_raw_1s_trades_6M.csv', index=False)
    
    if len(df) > 0:
        gross_pnl = df['gross_usd'].sum()
        net_pnl = df['net_usd'].sum()
        gross_profit = df[df['gross_usd'] > 0]['gross_usd'].sum()
        gross_loss = abs(df[df['gross_usd'] < 0]['gross_usd'].sum())
        wr = (gross_profit / gross_loss) - 1 if gross_loss != 0 else float('inf')
        
        print("\n=== TRUE 1s RAW PHYSICS (NO EXITS) ===")
        print(f"Total Trades: {len(df)}")
        print(f"Gross PnL:    ${gross_pnl:,.2f}")
        print(f"Friction:     ${len(df) * 2.5:,.2f}")
        print(f"Net PnL:      ${net_pnl:,.2f}")
        print(f"Win Rate:     {wr:.3f}")
        print(f"Avg MFE:      ${df['mfe_usd'].mean():.2f}")
        print(f"Avg MAE:      ${df['mae_usd'].mean():.2f}")
    else:
        print("No trades generated.")

if __name__ == '__main__':
    main()
