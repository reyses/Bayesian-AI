"""Load trained scenario LSTM and produce softmax probabilities on OOS set.
Saves P(LONG) per trade for threshold sweep in forward pass.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from tools.scenario_lstm_train import ScenarioLSTM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--seq-npz', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = torch.device('cuda')
    ck = torch.load(args.ckpt, weights_only=False)
    cfg = ck['config']
    model = ScenarioLSTM(n_feat=cfg['n_feat'], hidden=cfg['hidden'],
                         layers=cfg['layers'], dropout=cfg['dropout']).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()

    z = np.load(args.seq_npz, allow_pickle=True)
    X = torch.from_numpy(z['X']).float()

    print(f'Predicting on {len(X)} sequences...')
    p_long = []
    p_dur = []
    p_spd = []
    p_traj = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i+256].to(device)
            out = model(xb)
            p_long.append(torch.softmax(out['dir'], dim=1)[:, 1].cpu().numpy())
            p_dur.append(torch.softmax(out['dur'], dim=1).cpu().numpy())
            p_spd.append(torch.softmax(out['spd'], dim=1).cpu().numpy())
            p_traj.append(torch.softmax(out['traj'], dim=1).cpu().numpy())

    p_long = np.concatenate(p_long)
    p_dur = np.concatenate(p_dur)
    p_spd = np.concatenate(p_spd)
    p_traj = np.concatenate(p_traj)
    print(f'P(LONG): min={p_long.min():.3f} max={p_long.max():.3f} mean={p_long.mean():.3f}')

    np.savez_compressed(
        args.out,
        oracle_idx=z['oracle_idx'],
        p_long=p_long,
        p_dur=p_dur, p_spd=p_spd, p_traj=p_traj,
        y_dir_true=z['y_dir'], y_dur_true=z['y_dur'],
        y_spd_true=z['y_spd'], y_traj_true=z['y_traj'],
    )
    print(f'Wrote: {args.out}')


if __name__ == '__main__':
    main()
