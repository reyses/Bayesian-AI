"""
Visual Shape CNN — classifies level touches from candlestick images.

Renders 20-bar windows around each level touch as a small candlestick
image, then trains a 2D CNN to classify: reversal / breakout / plateau / bounce.

Parallel to the numerical shape analysis — compare results to see if
visual patterns capture something the 22D features miss.

Usage:
  python -m tools.visual_shape_cnn --tf 1h
  python -m tools.visual_shape_cnn --tf 1h --render-only  (just build images)
  python -m tools.visual_shape_cnn --tf 1h --train-only   (use cached images)
"""
import argparse
import gc
import glob
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import io
import pickle

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS = 'DATA/ATLAS'
TICK = 0.25
# Bars before and after touch — sized to capture ONE MOVE, not a full week
# Each move at any TF is typically 3-5 bars. Show 3 bars context + touch + 3 bars outcome.
TF_WINDOW = {
    '1D': 3,     # 7 bars
    '4h': 3,     # 7 bars = ~28 hours (one move)
    '1h': 3,     # 7 bars = ~7 hours
    '15m': 5,    # 11 bars = ~2.5 hours
    '5m': 5,     # 11 bars = ~55 minutes
    '1m': 5,     # 11 bars = ~11 minutes
}
TARGET_BAR_HEIGHT = 100  # median bar gets ~100px of vertical space
MAX_IMG_HEIGHT = 1200    # cap height so images stay reasonable
PIXELS_PER_BAR = 12      # horizontal: body width in pixels
WICK_WIDTH = 5           # wick width in pixels
# Image height is DATA-DRIVEN: scaled so median bar = TARGET_BAR_HEIGHT pixels
# Capped at MAX_IMG_HEIGHT — candles scale to fill the frame
# CNN handles variable-size inputs via AdaptiveAvgPool
CLASS_MAP = {'reversal': 0, 'breakout': 1, 'plateau': 2, 'bounce': 3}
CLASS_NAMES = ['reversal', 'breakout', 'plateau', 'bounce']


def render_candle_panel(ax, opens, highs, lows, closes, level_price, title=''):
    """Render candlesticks on a matplotlib axis.

    Body = PIXELS_PER_BAR wide (green/red)
    Wick = WICK_WIDTH wide (gray)
    """
    n = len(closes)
    x = np.arange(n)
    body_w = 0.8   # relative to bar spacing (matplotlib units)
    wick_w = body_w * (WICK_WIDTH / PIXELS_PER_BAR)  # wick proportional to body

    for i in range(n):
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        color = '#26A69A' if c >= o else '#EF5350'
        # Wick (full high-low range)
        ax.bar(x[i], h - l, bottom=l, width=wick_w, color='#888888', edgecolor='none')
        # Body (open-close)
        body = max(abs(c - o), TICK)
        ax.bar(x[i], body, bottom=min(o, c), width=body_w, color=color, edgecolor='none')

    ax.axhline(y=level_price, color='white', linewidth=0.5, alpha=0.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.axis('off')
    if title:
        ax.set_title(title, color='white', fontsize=6, pad=1)


def render_candle_image(opens, highs, lows, closes, level_price):
    """Render a single-TF candlestick window as RGB numpy array.

    Image size is DATA-DRIVEN:
      Height = scaled so median bar = TARGET_BAR_HEIGHT pixels
      Width = n_bars × PIXELS_PER_BAR
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(closes)
    bar_ranges = highs - lows
    median_bar = max(np.median(bar_ranges), TICK)  # avoid zero
    px_per_point = TARGET_BAR_HEIGHT / median_bar
    window_range = max(highs) - min(lows)
    img_h = min(MAX_IMG_HEIGHT, max(100, int(window_range * px_per_point)))
    img_w = max(100, n * PIXELS_PER_BAR)

    dpi = 32
    fig, ax = plt.subplots(1, 1, figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    render_candle_panel(ax, opens, highs, lows, closes, level_price)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # RGBA -> RGB
    plt.close(fig)

    return img


def render_multi_tf_image(touch_ts, level_price, tf_data, img_w=256, img_h=256):
    """Render 3 TFs stacked vertically around a level touch.

    tf_data: dict of tf -> DataFrame with OHLCV
    Shows the same moment at 3 zoom levels.
    Returns (img_h, img_w, 3) RGB array.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tfs = list(tf_data.keys())[:3]  # max 3 panels
    n_panels = len(tfs)

    fig, axes = plt.subplots(n_panels, 1, figsize=(img_w / 32, img_h / 32), dpi=32)
    fig.patch.set_facecolor('black')
    if n_panels == 1:
        axes = [axes]

    for idx, tf in enumerate(tfs):
        ax = axes[idx]
        ax.set_facecolor('black')
        df_tf = tf_data[tf]

        if len(df_tf) == 0:
            continue

        render_candle_panel(ax, df_tf['open'].values, df_tf['high'].values,
                           df_tf['low'].values, df_tf['close'].values,
                           level_price, title=tf)

    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, hspace=0.1)

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)

    if img.shape[0] != img_h or img.shape[1] != img_w:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img)
        pil_img = pil_img.resize((img_w, img_h), PILImage.BILINEAR)
        img = np.array(pil_img)

    return img


def build_image_dataset(tf, months=None):
    """Build image dataset from level touches. Returns images, labels, metadata, context_features."""
    from core_v2.statistical_field_engine import StatisticalFieldEngine
    from training.train_trade_cnn import extract_features_13d
    from tools.level_shapes import get_levels, classify_touch

    window = TF_WINDOW[tf]

    level_files = sorted(glob.glob('DATA/levels/levels_*.json'))
    if months is None:
        months = sorted(set(json.load(open(f))['date'][:7] for f in level_files))

    images = []
    labels = []
    metadata = []
    context_features = []  # 13D per touch

    for month in tqdm(months, desc=f"Rendering {tf}"):
        levels = get_levels(month)
        if not levels or len(levels) < 2:
            continue

        # Load data
        month_str = month.replace('-', '_')
        path = os.path.join(ATLAS, tf, f'{month_str}.parquet')
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
        if len(df) < 2 * window + 5:
            continue

        # Compute 13D features for context stream
        sfe = StatisticalFieldEngine()
        states = sfe.batch_compute_states(df)
        feats_13d = extract_features_13d(states, df)
        del states, sfe; gc.collect()

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        n = len(closes)

        level_prices = [(l['price'], l['type']) for l in levels]

        for lp, ltype in level_prices:
            for i in range(window, n - window):
                # Touch detection
                touched = (lows[i] <= lp + 15 and highs[i] >= lp - 15)
                if not touched:
                    continue

                classification = classify_touch(closes, highs, lows, i, lp, ltype)
                if classification is None or classification not in CLASS_MAP:
                    continue

                # Extract window
                sl = slice(i - window, i + window + 1)
                w_opens = opens[sl]
                w_highs = highs[sl]
                w_lows = lows[sl]
                w_closes = closes[sl]

                if len(w_closes) != 2 * window + 1:
                    continue

                # Render image
                img = render_candle_image(w_opens, w_highs, w_lows, w_closes, lp)
                images.append(img)
                labels.append(CLASS_MAP[classification])
                context_features.append(feats_13d[i].copy())
                metadata.append({
                    'month': month, 'bar': i, 'level': lp,
                    'type': ltype, 'class': classification,
                    'opens': w_opens.tolist(), 'highs': w_highs.tolist(),
                    'lows': w_lows.tolist(), 'closes': w_closes.tolist(),
                })

        del df, feats_13d; gc.collect()

    return images, np.array(labels), metadata, np.array(context_features)


class CandleImageDataset(Dataset):
    """Native-resolution images + optional 13D context features."""

    def __init__(self, images, labels, context_features=None):
        self.raw_images = images  # list of (H, W, 3) arrays (variable size)
        self.labels = torch.LongTensor(labels)
        self.context = context_features  # list of (13,) arrays or None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.raw_images[idx]
        arr = img.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(arr).permute(2, 0, 1)  # (3, H, W)
        if self.context is not None:
            ctx = torch.FloatTensor(self.context[idx])
        else:
            ctx = torch.zeros(N_CONTEXT_FEATURES)
        return tensor, self.labels[idx], ctx



N_CONTEXT_FEATURES = 13  # 13D numerical context alongside image


class VisualShapeCNN(nn.Module):
    """Two-stream CNN: image (Conv2d) + numerical context (MLP), merged at classifier.

    Stream 1 — Image: Conv2d(3→16→32→64) + AdaptiveAvgPool → 64*4*4 = 1024D
    Stream 2 — Context: 13D → 32D → 32D (numerical features: dmi, velocity, etc.)
    Merge: concat(1024D, 32D) = 1056D → 64D → n_classes
    """

    def __init__(self, n_classes=4, n_context=N_CONTEXT_FEATURES):
        super().__init__()
        self.n_context = n_context

        # Stream 1: Image backbone — last conv output stored for Grad-CAM
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.img_flat_dim = 64 * 4 * 4  # 1024

        # Stream 2: Numerical context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(n_context, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Merged classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.img_flat_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

        # For backwards compat
        self.features = nn.Sequential(self.conv1, self.conv2, self.conv3, self.pool)
        _total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VisualShapeCNN] {_total:,} params | image: variable RGB + {n_context}D context | output: {n_classes} classes")

    def forward(self, x, context=None):
        # Stream 1: Image
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        if h.requires_grad:
            h.retain_grad()
        self._last_conv = h
        h = self.pool(h)
        img_flat = h.view(h.size(0), -1)  # (B, 1024)

        # Stream 2: Context (or zeros if not provided)
        if context is not None:
            ctx = self.context_encoder(context)  # (B, 32)
        else:
            ctx = torch.zeros(img_flat.size(0), 32, device=img_flat.device)

        # Merge
        merged = torch.cat([img_flat, ctx], dim=1)  # (B, 1056)
        return self.classifier(merged)


def train_visual_cnn(images, labels, context_features=None, n_epochs=50, val_split=0.2):
    """Train VisualShapeCNN on rendered candlestick images + 13D context."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split
    n = len(images)
    perm = np.random.RandomState(42).permutation(n)
    n_val = int(n * val_split)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_images = [images[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    train_ctx = context_features[train_idx] if context_features is not None else None
    val_ctx = context_features[val_idx] if context_features is not None else None
    train_ds = CandleImageDataset(train_images, labels[train_idx], train_ctx)
    val_ds = CandleImageDataset(val_images, labels[val_idx], val_ctx)

    # batch_size=1: images are variable-size, can't stack into batches
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Class balance
    class_counts = np.bincount(labels[train_idx], minlength=4)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * 4
    weight_tensor = torch.FloatTensor(class_weights).to(device)

    model = VisualShapeCNN(n_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    best_acc = 0
    best_state = None

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training"):
        model.train()
        for x, y, ctx in train_dl:
            x, y, ctx = x.to(device), y.to(device), ctx.to(device)
            pred = model(x, context=ctx)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            correct = 0
            total = 0
            per_class = {c: {'correct': 0, 'total': 0} for c in range(4)}
            with torch.no_grad():
                for x, y, ctx in val_dl:
                    x, y, ctx = x.to(device), y.to(device), ctx.to(device)
                    pred = model(x, context=ctx)
                    predicted = pred.argmax(dim=1)
                    correct += (predicted == y).sum().item()
                    total += len(y)
                    for c in range(4):
                        mask = y == c
                        per_class[c]['total'] += mask.sum().item()
                        per_class[c]['correct'] += (predicted[mask] == c).sum().item()

            acc = correct / total * 100
            pc_str = ' '.join([f'{CLASS_NAMES[c][:3]}={per_class[c]["correct"]}/{per_class[c]["total"]}'
                               for c in range(4)])
            tqdm.write(f"  E{epoch}: acc={acc:.1f}% | {pc_str}")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  Best accuracy: {best_acc:.1f}%")
    return model, best_acc


def grad_cam(model, tensor, pred_class, context=None):
    """Compute Grad-CAM heatmap for the predicted class.

    Returns heatmap as (H, W) numpy array in [0, 1].
    """
    model.eval()
    tensor.requires_grad_(True)

    logits = model(tensor, context=context)
    model.zero_grad()
    logits[0, pred_class].backward()

    # Gradients of last conv layer
    conv_out = model._last_conv  # (1, 64, H', W')
    grads = conv_out.grad        # same shape

    # Channel weights = global average of gradients per channel
    weights = grads.mean(dim=[2, 3], keepdim=True)  # (1, 64, 1, 1)
    cam = (weights * conv_out).sum(dim=1, keepdim=True)  # (1, 1, H', W')
    cam = torch.relu(cam).squeeze().cpu().detach().numpy()

    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def overlay_heatmap(pil_img, cam, alpha=0.4):
    """Overlay Grad-CAM heatmap on PIL image. Returns blended PIL image."""
    from PIL import Image as PILImage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm

    # Resize CAM to image size
    cam_resized = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
        (pil_img.width, pil_img.height), PILImage.BILINEAR)) / 255.0

    # Apply colormap (jet: blue=cold, red=hot)
    colormap = cm.jet(cam_resized)[:, :, :3]  # (H, W, 3) in [0, 1]
    heatmap = (colormap * 255).astype(np.uint8)

    # Blend: original * (1-alpha) + heatmap * alpha
    original = np.array(pil_img).astype(np.float32)
    blended = original * (1 - alpha) + heatmap.astype(np.float32) * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return PILImage.fromarray(blended)


def draw_reference_lines(draw, img_w, img_h, meta, tf):
    """Draw interpretive reference lines on an audit image.

    Lines drawn:
      - Level line (yellow, solid thick) — the S/R level
      - Touch bar (white vertical thick) — center bar where touch occurred
      - Pre-trend line (cyan thick) — linear regression before touch
      - Post-trend line (magenta thick) — linear regression after touch
      - S/R label at the level line
    """
    if not meta or 'closes' not in meta:
        return

    closes = np.array(meta['closes'])
    highs = np.array(meta['highs'])
    lows = np.array(meta['lows'])
    level = meta['level']
    n = len(closes)
    window = n // 2  # center bar index

    # Map price -> pixel Y (top = max price, bottom = min price)
    price_min = min(lows)
    price_max = max(highs)
    price_range = price_max - price_min
    if price_range < TICK:
        return

    chart_h = img_h

    def price_to_y(p):
        return int((1 - (p - price_min) / price_range) * chart_h)

    def bar_to_x(b):
        return int((b + 0.5) * img_w / n)

    # Line widths scale with image size so they're always visible
    thick = max(4, img_w // 60)
    thin = max(2, thick // 2)

    # 1. Level line — bright yellow, double-outline for visibility
    ly = price_to_y(level)
    if 0 <= ly <= img_h:
        draw.line([(0, ly), (img_w, ly)], fill='#000000', width=thick + 2)  # black outline
        draw.line([(0, ly), (img_w, ly)], fill='#FFD700', width=thick)
        label = meta.get('type', 'unknown').upper()
        # Label with black background for readability
        draw.rectangle([img_w - 100, ly + 5, img_w - 2, ly + 22], fill='#000000')
        draw.text((img_w - 98, ly + 6), label, fill='#FFD700')

    # 2. Touch bar — white vertical with black outline
    tx = bar_to_x(window)
    draw.line([(tx, 0), (tx, img_h)], fill='#000000', width=thin + 2)
    draw.line([(tx, 0), (tx, img_h)], fill='#FFFFFF', width=thin)
    draw.rectangle([tx + 4, 30, tx + 56, 46], fill='#000000')
    draw.text((tx + 6, 32), 'TOUCH', fill='#FFFFFF')

    # 3. Pre-trend line — cyan with outline
    pre_closes = closes[:window]
    if len(pre_closes) >= 3:
        x_pre = np.arange(len(pre_closes))
        coeffs = np.polyfit(x_pre, pre_closes, 1)
        p_start = coeffs[0] * 0 + coeffs[1]
        p_end = coeffs[0] * (len(pre_closes) - 1) + coeffs[1]
        y_start = price_to_y(p_start)
        y_end = price_to_y(p_end)
        x_start = bar_to_x(0)
        x_end = bar_to_x(window - 1)
        draw.line([(x_start, y_start), (x_end, y_end)], fill='#000000', width=thick + 2)
        draw.line([(x_start, y_start), (x_end, y_end)], fill='#00FFFF', width=thick)
        slope_ticks = (p_end - p_start) / TICK
        slope_dir = 'UP' if coeffs[0] > 0 else 'DN'
        lbl = f'{slope_dir} {abs(slope_ticks):.0f}t'
        lbl_y = max(50, min(y_start, y_end) - 20)
        draw.rectangle([x_start + 2, lbl_y, x_start + len(lbl) * 8 + 6, lbl_y + 16], fill='#000000')
        draw.text((x_start + 4, lbl_y + 1), lbl, fill='#00FFFF')

    # 4. Post-trend line — magenta with outline
    post_closes = closes[window + 1:]
    if len(post_closes) >= 3:
        x_post = np.arange(len(post_closes))
        coeffs = np.polyfit(x_post, post_closes, 1)
        p_start = coeffs[0] * 0 + coeffs[1]
        p_end = coeffs[0] * (len(post_closes) - 1) + coeffs[1]
        y_start = price_to_y(p_start)
        y_end = price_to_y(p_end)
        x_start = bar_to_x(window + 1)
        x_end = bar_to_x(n - 1)
        draw.line([(x_start, y_start), (x_end, y_end)], fill='#000000', width=thick + 2)
        draw.line([(x_start, y_start), (x_end, y_end)], fill='#FF00FF', width=thick)
        slope_ticks = (p_end - p_start) / TICK
        slope_dir = 'UP' if coeffs[0] > 0 else 'DN'
        lbl = f'{slope_dir} {abs(slope_ticks):.0f}t'
        lbl_y = max(50, min(y_start, y_end) - 20)
        draw.rectangle([x_end - len(lbl) * 8 - 6, lbl_y, x_end - 2, lbl_y + 16], fill='#000000')
        draw.text((x_end - len(lbl) * 8 - 4, lbl_y + 1), lbl, fill='#FF00FF')


def audit_predictions(model, images, labels, metadata, out_dir,
                      context_features=None, max_per_class=20):
    """Save annotated images with Grad-CAM heatmap + reference lines."""
    from PIL import Image as PILImage, ImageDraw

    device = next(model.parameters()).device
    model.eval()

    audit_dir = os.path.join(out_dir, 'audit')
    os.makedirs(audit_dir, exist_ok=True)
    for sub in ['correct', 'wrong']:
        os.makedirs(os.path.join(audit_dir, sub), exist_ok=True)

    counts = {'correct': 0, 'wrong': 0}
    class_counts = {c: 0 for c in range(4)}

    for i in tqdm(range(len(images)), desc="Auditing"):
        label = int(labels[i])
        if class_counts[label] >= max_per_class:
            continue

        img = images[i]
        arr = img.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        # Context features
        if context_features is not None:
            ctx = torch.FloatTensor(context_features[i]).unsqueeze(0).to(device)
        else:
            ctx = None

        # Forward + Grad-CAM
        logits = model(tensor, context=ctx)
        probs = torch.softmax(logits, dim=1)[0].cpu().detach().numpy()
        pred = int(logits.argmax(dim=1).item())
        conf = probs[pred] * 100

        cam = grad_cam(model, tensor, pred, context=ctx)

        # Build annotated image with heatmap
        pil = PILImage.fromarray(img)
        pil = overlay_heatmap(pil, cam, alpha=0.4)
        draw = ImageDraw.Draw(pil)

        truth_name = CLASS_NAMES[label]
        pred_name = CLASS_NAMES[pred]
        is_correct = pred == label

        # Header: truth vs prediction (drawn first as background)
        header = f"TRUTH: {truth_name} | PRED: {pred_name} ({conf:.0f}%)"
        prob_str = ' '.join([f"{CLASS_NAMES[c][:3]}:{probs[c]:.0%}" for c in range(4)])
        bar_h = 28
        draw.rectangle([0, 0, pil.width, bar_h], fill='black')
        text_color = '#26A69A' if is_correct else '#EF5350'
        draw.text((4, 2), header, fill=text_color)
        draw.text((4, 14), prob_str, fill='#AAAAAA')

        # Footer: metadata
        meta = metadata[i] if metadata and i < len(metadata) else None
        if meta:
            footer = f"{meta['month']} bar:{meta['bar']} lvl:{meta['level']} ({meta['type']})"
            draw.rectangle([0, pil.height - 16, pil.width, pil.height], fill='black')
            draw.text((4, pil.height - 14), footer, fill='#888888')

        # Reference lines ON TOP of everything (level, touch bar, pre/post trend)
        tf = out_dir.split('_')[-1] if '_' in out_dir else '4h'
        draw_reference_lines(draw, pil.width, pil.height, meta, tf)

        sub = 'correct' if is_correct else 'wrong'
        fname = f"{truth_name}_{i:04d}_pred_{pred_name}_{conf:.0f}pct.png"
        pil.save(os.path.join(audit_dir, sub, fname))

        counts[sub] += 1
        class_counts[label] += 1

    print(f"\n  Audit: {counts['correct']} correct, {counts['wrong']} wrong")
    print(f"  Saved to {audit_dir}/correct/ and {audit_dir}/wrong/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='1h', choices=['1D', '4h', '1h', '15m', '5m', '1m'])
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true', help='Load trained model and save annotated images')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--audit-n', type=int, default=20, help='Max images per class to audit')
    parser.add_argument('--months', default=None, help='Comma-separated YYYY-MM (e.g. 2025-06)')
    args = parser.parse_args()

    ckpt_dir = f'checkpoints/visual_shape_{args.tf}'
    os.makedirs(ckpt_dir, exist_ok=True)

    img_cache = os.path.join(ckpt_dir, 'images.pkl')
    lbl_cache = os.path.join(ckpt_dir, 'labels.npy')

    ctx_cache = os.path.join(ckpt_dir, 'context_features.npy')

    if args.train_only and os.path.exists(img_cache):
        print("Loading cached images...")
        with open(img_cache, 'rb') as f:
            images = pickle.load(f)
        labels = np.load(lbl_cache)
        context_features = np.load(ctx_cache) if os.path.exists(ctx_cache) else None
    else:
        print(f"Building image dataset from {args.tf} level touches...")
        months = args.months.split(',') if args.months else None
        images, labels, metadata, context_features = build_image_dataset(args.tf, months=months)
        with open(img_cache, 'wb') as f:
            pickle.dump(images, f)
        np.save(lbl_cache, labels)
        meta_cache = os.path.join(ckpt_dir, 'metadata.pkl')
        with open(meta_cache, 'wb') as f:
            pickle.dump(metadata, f)
        ctx_cache = os.path.join(ckpt_dir, 'context_features.npy')
        np.save(ctx_cache, context_features)
        print(f"  Saved: {img_cache} ({len(images)} images, {context_features.shape[1]}D context)")

    print(f"\nDataset: {len(images)} images (native resolution, no compression)")
    for c in range(4):
        print(f"  {CLASS_NAMES[c]}: {(labels == c).sum()}")

    # Load metadata if available (needed for audit annotations)
    meta_cache = os.path.join(ckpt_dir, 'metadata.pkl')
    metadata = None
    if os.path.exists(meta_cache):
        with open(meta_cache, 'rb') as f:
            metadata = pickle.load(f)

    # Load context features if not already loaded (for audit-only / render-only paths)
    try:
        context_features
    except NameError:
        context_features = np.load(ctx_cache) if os.path.exists(ctx_cache) else None

    if args.render_only:
        # Save a few sample images
        for c in range(4):
            idx = np.where(labels == c)[0]
            if len(idx) > 0:
                sample = images[idx[0]]
                from PIL import Image as PILImage
                PILImage.fromarray(sample).save(
                    os.path.join(ckpt_dir, f'sample_{CLASS_NAMES[c]}.png'))
        print(f"  Samples saved to {ckpt_dir}/")
        return

    if args.audit_only:
        # Load existing model and audit
        model_path = os.path.join(ckpt_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            print(f"  No model found at {model_path}")
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VisualShapeCNN(n_classes=4).to(device)
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        audit_predictions(model, images, labels, metadata, ckpt_dir,
                          context_features=context_features, max_per_class=args.audit_n)
        return

    print(f"\nTraining VisualShapeCNN ({args.epochs} epochs)...")
    model, best_acc = train_visual_cnn(images, labels, context_features=context_features,
                                        n_epochs=args.epochs)

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'tf': args.tf,
        'best_acc': best_acc,
        'n_images': len(images),
        'class_counts': {CLASS_NAMES[c]: int((labels == c).sum()) for c in range(4)},
    }, os.path.join(ckpt_dir, 'best_model.pt'))
    print(f"  Saved: {ckpt_dir}/best_model.pt")

    # Auto-audit after training
    print(f"\nRunning audit...")
    audit_predictions(model, images, labels, metadata, ckpt_dir,
                      context_features=context_features, max_per_class=args.audit_n)


if __name__ == '__main__':
    main()
