"""LSTM multi-head scenario classifier — train on IS, eval on OOS.

Per user 2026-05-16: predict 4 scenario buckets (direction, duration, speed,
trajectory) from lead-in sequence (K=60 bars x N=30 features).

Architecture (CUDA-only per CLAUDE.md):
  Input:  (batch, K, N) float32 z-scored
  Trunk:  LSTM(input=N, hidden=64, layers=1, dropout=0.3, batch_first=True)
  Pool:   last hidden state (batch, 64)
  Heads:
    dir_head:  Linear(64, 2)   — LONG/SHORT
    dur_head:  Linear(64, 4)
    spd_head:  Linear(64, 4)
    traj_head: Linear(64, 4)
  Loss: sum of per-head cross-entropy with inverse-frequency class weights

Train:
  - 80/20 train/val split (seed 42, parity with direction classifier)
  - Adam lr=1e-3 weight_decay=1e-4
  - Batch 128, max 50 epochs, early stop patience 5 on val total loss
  - Save best model + predictions

Eval (--eval-oos):
  - Load best checkpoint
  - Predict on OOS sequence dataset
  - Report per-head accuracy + confusion matrix IS-test vs OOS
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ScenarioLSTM(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=hidden, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head_dir = nn.Linear(hidden, 2)
        self.head_dur = nn.Linear(hidden, 4)
        self.head_spd = nn.Linear(hidden, 4)
        self.head_traj = nn.Linear(hidden, 4)

    def forward(self, x):
        # x: (B, K, N)
        out, (h, c) = self.lstm(x)
        # last layer's last step
        z = h[-1]            # (B, hidden)
        z = self.dropout(z)
        return {
            'dir': self.head_dir(z),
            'dur': self.head_dur(z),
            'spd': self.head_spd(z),
            'traj': self.head_traj(z),
        }


def class_weights(y: np.ndarray, n_classes: int, device) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.from_numpy(w).to(device)


def evaluate(model, loader, device):
    model.eval()
    n = 0
    correct = {'dir': 0, 'dur': 0, 'spd': 0, 'traj': 0}
    losses = {'dir': 0.0, 'dur': 0.0, 'spd': 0.0, 'traj': 0.0}
    preds = {k: [] for k in correct}
    trues = {k: [] for k in correct}
    with torch.no_grad():
        for X, yd, ydu, ys, yt in loader:
            X = X.to(device); yd = yd.to(device); ydu = ydu.to(device)
            ys = ys.to(device); yt = yt.to(device)
            out = model(X)
            for k, y in (('dir', yd), ('dur', ydu), ('spd', ys), ('traj', yt)):
                p = out[k].argmax(1)
                correct[k] += (p == y).sum().item()
                preds[k].append(p.cpu().numpy())
                trues[k].append(y.cpu().numpy())
            n += X.size(0)
    accs = {k: correct[k]/n for k in correct}
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    trues = {k: np.concatenate(v) for k, v in trues.items()}
    return accs, preds, trues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-npz', help='IS sequence dataset')
    ap.add_argument('--oos-npz', help='OOS sequence dataset for eval')
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    ap.add_argument('--name', default='scenario_lstm')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-only', action='store_true',
                    help='Skip training, just eval (requires existing checkpoint)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; PyTorch {torch.__version__}')
    assert device.type == 'cuda', 'CUDA-only per CLAUDE.md'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'\nLoading IS: {args.train_npz}')
    is_z = np.load(args.train_npz, allow_pickle=True)
    X_is = is_z['X']  # (N, K, F)
    y_dir = is_z['y_dir'].astype(np.int64)
    y_dur = is_z['y_dur'].astype(np.int64)
    y_spd = is_z['y_spd'].astype(np.int64)
    y_traj = is_z['y_traj'].astype(np.int64)
    print(f'  X: {X_is.shape}  features: {list(is_z["feature_names"][:5])} ...')

    # Train/val split (random, seed 42 — same as direction classifier)
    n = X_is.shape[0]
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_val = int(0.2 * n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def to_tensors(ix):
        return (
            torch.from_numpy(X_is[ix]).float(),
            torch.from_numpy(y_dir[ix]),
            torch.from_numpy(y_dur[ix]),
            torch.from_numpy(y_spd[ix]),
            torch.from_numpy(y_traj[ix]),
        )

    ds_tr = TensorDataset(*to_tensors(tr_idx))
    ds_va = TensorDataset(*to_tensors(val_idx))
    loader_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=False)
    loader_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, drop_last=False)
    print(f'  Train: {len(tr_idx)}    Val (test): {len(val_idx)}')

    # Class weights from train split
    w_dir = class_weights(y_dir[tr_idx], 2, device)
    w_dur = class_weights(y_dur[tr_idx], 4, device)
    w_spd = class_weights(y_spd[tr_idx], 4, device)
    w_traj = class_weights(y_traj[tr_idx], 4, device)
    print(f'  Class weights — dir: {w_dir.tolist()}')
    print(f'  Class weights — dur: {w_dur.tolist()}')
    print(f'  Class weights — spd: {w_spd.tolist()}')
    print(f'  Class weights — traj: {w_traj.tolist()}')

    # Model
    n_feat = X_is.shape[2]
    model = ScenarioLSTM(n_feat=n_feat, hidden=args.hidden,
                         layers=args.layers, dropout=args.dropout).to(device)
    print(f'\nModel: ScenarioLSTM(n_feat={n_feat}, hidden={args.hidden}, '
          f'layers={args.layers}, dropout={args.dropout})')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ckpt_path = out_dir / f'{args.name}.pt'

    if not args.eval_only:
        print(f'\n--- Training ---')
        best_val = float('inf')
        bad_epochs = 0
        history = []
        for epoch in range(args.epochs):
            t_e0 = time.time()
            model.train()
            tot = 0; loss_sum = 0.0
            for X, yd, ydu, ys, yt in loader_tr:
                X = X.to(device); yd = yd.to(device); ydu = ydu.to(device)
                ys = ys.to(device); yt = yt.to(device)
                out = model(X)
                ld = nn.functional.cross_entropy(out['dir'], yd, weight=w_dir)
                lu = nn.functional.cross_entropy(out['dur'], ydu, weight=w_dur)
                ls = nn.functional.cross_entropy(out['spd'], ys, weight=w_spd)
                lt = nn.functional.cross_entropy(out['traj'], yt, weight=w_traj)
                loss = ld + lu + ls + lt
                opt.zero_grad(); loss.backward(); opt.step()
                tot += X.size(0); loss_sum += loss.item() * X.size(0)
            train_loss = loss_sum / tot

            # Val
            model.eval()
            v_tot = 0; v_loss_sum = 0.0
            with torch.no_grad():
                for X, yd, ydu, ys, yt in loader_va:
                    X = X.to(device); yd = yd.to(device); ydu = ydu.to(device)
                    ys = ys.to(device); yt = yt.to(device)
                    out = model(X)
                    ld = nn.functional.cross_entropy(out['dir'], yd, weight=w_dir)
                    lu = nn.functional.cross_entropy(out['dur'], ydu, weight=w_dur)
                    ls = nn.functional.cross_entropy(out['spd'], ys, weight=w_spd)
                    lt = nn.functional.cross_entropy(out['traj'], yt, weight=w_traj)
                    loss = ld + lu + ls + lt
                    v_tot += X.size(0); v_loss_sum += loss.item() * X.size(0)
            val_loss = v_loss_sum / v_tot
            val_accs, _, _ = evaluate(model, loader_va, device)

            history.append({
                'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                'val_acc_dir': val_accs['dir'], 'val_acc_dur': val_accs['dur'],
                'val_acc_spd': val_accs['spd'], 'val_acc_traj': val_accs['traj'],
                'elapsed_s': time.time()-t_e0,
            })
            print(f'  ep{epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  '
                  f'acc dir={val_accs["dir"]:.3f} dur={val_accs["dur"]:.3f} '
                  f'spd={val_accs["spd"]:.3f} traj={val_accs["traj"]:.3f}  '
                  f'({time.time()-t_e0:.1f}s)', flush=True)

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                bad_epochs = 0
                torch.save({
                    'model_state': model.state_dict(),
                    'config': {
                        'n_feat': n_feat, 'hidden': args.hidden,
                        'layers': args.layers, 'dropout': args.dropout,
                    },
                    'epoch': epoch, 'val_loss': val_loss,
                    'val_accs': val_accs,
                }, ckpt_path)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f'  Early stopping at epoch {epoch} (best val_loss {best_val:.4f})')
                    break

        # Save history
        import pandas as pd
        pd.DataFrame(history).to_csv(out_dir / f'{args.name}_history.csv', index=False)
        print(f'\nSaved best checkpoint: {ckpt_path}')

    # Load best for eval
    print(f'\n--- Loading best checkpoint for evaluation ---')
    ck = torch.load(ckpt_path, weights_only=False)
    cfg = ck['config']
    model = ScenarioLSTM(n_feat=cfg['n_feat'], hidden=cfg['hidden'],
                         layers=cfg['layers'], dropout=cfg['dropout']).to(device)
    model.load_state_dict(ck['model_state'])
    print(f'  Loaded epoch {ck["epoch"]}, val_loss {ck["val_loss"]:.4f}')

    # Eval on IS-val (test set)
    print(f'\n=== IS test (validation) set ===')
    is_accs, is_preds, is_trues = evaluate(model, loader_va, device)
    for k in ('dir', 'dur', 'spd', 'traj'):
        print(f'  {k} accuracy: {is_accs[k]:.4f}    '
              f'baseline (most-common): {np.bincount(is_trues[k]).max()/len(is_trues[k]):.4f}')

    # Eval on OOS
    if args.oos_npz:
        print(f'\nLoading OOS: {args.oos_npz}')
        oos_z = np.load(args.oos_npz, allow_pickle=True)
        X_oos = oos_z['X']
        y_dir_o = oos_z['y_dir'].astype(np.int64)
        y_dur_o = oos_z['y_dur'].astype(np.int64)
        y_spd_o = oos_z['y_spd'].astype(np.int64)
        y_traj_o = oos_z['y_traj'].astype(np.int64)
        print(f'  X: {X_oos.shape}')
        ds_oos = TensorDataset(
            torch.from_numpy(X_oos).float(),
            torch.from_numpy(y_dir_o),
            torch.from_numpy(y_dur_o),
            torch.from_numpy(y_spd_o),
            torch.from_numpy(y_traj_o),
        )
        loader_oos = DataLoader(ds_oos, batch_size=args.batch, shuffle=False)
        print(f'\n=== OOS (2026) set ===')
        oos_accs, oos_preds, oos_trues = evaluate(model, loader_oos, device)
        for k in ('dir', 'dur', 'spd', 'traj'):
            print(f'  {k} accuracy: {oos_accs[k]:.4f}    '
                  f'baseline (most-common): {np.bincount(oos_trues[k]).max()/len(oos_trues[k]):.4f}    '
                  f'IS-OOS delta: {oos_accs[k]-is_accs[k]:+.4f}')

        # Save predictions
        np.savez_compressed(
            out_dir / f'{args.name}_OOS_predictions.npz',
            oracle_idx=oos_z['oracle_idx'],
            y_dir_true=y_dir_o, y_dir_pred=oos_preds['dir'],
            y_dur_true=y_dur_o, y_dur_pred=oos_preds['dur'],
            y_spd_true=y_spd_o, y_spd_pred=oos_preds['spd'],
            y_traj_true=y_traj_o, y_traj_pred=oos_preds['traj'],
            is_accs={k: float(v) for k, v in is_accs.items()},
            oos_accs={k: float(v) for k, v in oos_accs.items()},
        )

        # Confusion matrices (text)
        from sklearn.metrics import confusion_matrix
        print(f'\n=== OOS confusion matrices ===')
        for k in ('dir', 'dur', 'spd', 'traj'):
            cm = confusion_matrix(oos_trues[k], oos_preds[k])
            print(f'\n  {k}:')
            print(cm)

        # Summary JSON
        summary = {
            'is_test_acc': {k: float(v) for k, v in is_accs.items()},
            'oos_acc': {k: float(v) for k, v in oos_accs.items()},
            'config': {**cfg, 'lookback_bars': int(is_z['lookback_bars'][0])},
            'n_is_test': int(len(y_dir_o)),
            'n_oos': int(len(y_dir_o)),
        }
        with open(out_dir / f'{args.name}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'\nSaved summary: {out_dir / f"{args.name}_summary.json"}')


if __name__ == '__main__':
    main()
