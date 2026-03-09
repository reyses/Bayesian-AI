# Jules Spec: Waveform Seed Library Integration

## Goal

Bridge the offline waveform analysis (71.7% direction accuracy) into the live trading
pipeline. Deploy the 20-shape seed library + a trained GradientBoosting direction model
as new direction signals in the orchestrator forward pass and live engine.

**Current direction accuracy**: ~50.3% OOS
**Target**: 60-70%+ OOS direction accuracy via shape + stacked model signals

---

## Part 1: Extract SeedPrimitiveLibrary to Shared Module

**New file: `training/seed_library.py`**

Extract the `SeedPrimitiveLibrary` class from `tools/waveform_standalone.py` lines 1671-1762.
The class builds 20 normalized mathematical shape templates and classifies price trajectories
via Pearson correlation.

### Code

```python
"""Seed primitive library — 20 mathematical shape templates for trajectory classification."""

import pickle
import numpy as np


class SeedPrimitiveLibrary:
    """Library of 20 normalized seed shapes for trajectory classification.

    Shapes are grouped into three categories:
    1. Directional (8): LINEAR, EXPONENTIAL, LOGARITHMIC, STEP (UP/DOWN)
    2. Reversals (8): SYMMETRIC_V, ROUNDED_U, FRONT_SKEWED, BACK_SKEWED (UP/DOWN)
    3. Volatility (4): SINE_WAVE, DAMPED_OSCILLATOR, EXPAND_OSCILLATOR, FLATLINE
    """

    CORR_THRESHOLD = 0.75

    # shape_name -> guaranteed next-direction ('long'/'short'/None for ambiguous)
    DIRECTION_MAP = {
        'LINEAR_UP': 'long',          'LINEAR_DOWN': 'short',
        'EXPONENTIAL_UP': 'long',     'EXPONENTIAL_DOWN': 'short',
        'LOGARITHMIC_UP': 'long',     'LOGARITHMIC_DOWN': 'short',
        'STEP_UP': 'long',            'STEP_DOWN': 'short',
        'SYMMETRIC_V_UP': 'long',     'SYMMETRIC_V_DOWN': 'short',
        'ROUNDED_U_UP': 'long',       'ROUNDED_U_DOWN': 'short',
        'FRONT_SKEWED_UP': 'long',    'FRONT_SKEWED_DOWN': 'short',
        'BACK_SKEWED_UP': 'long',     'BACK_SKEWED_DOWN': 'short',
        'SINE_WAVE': None,
        'DAMPED_OSCILLATOR': None,
        'EXPAND_OSCILLATOR': None,
        'FLATLINE': None,
    }

    def __init__(self, N=16):
        self.N = N
        self.shapes = {}
        self._build(N)

    def _build(self, N):
        # --- COPY lines 1686-1722 from tools/waveform_standalone.py verbatim ---
        # Build all 20 shapes as normalized [0,1] numpy arrays of length N.
        # See waveform_standalone.py for the exact mathematical functions.
        x = np.linspace(0, 1, N)

        # Category 1: Directional (4 base × 2)
        def _norm(a):
            mn, mx = a.min(), a.max()
            return (a - mn) / (mx - mn) if mx > mn else np.zeros(N)

        self.shapes['LINEAR_UP']        = _norm(x)
        self.shapes['LINEAR_DOWN']      = _norm(-x)
        self.shapes['EXPONENTIAL_UP']   = _norm(x ** 2)
        self.shapes['EXPONENTIAL_DOWN'] = _norm(-(x ** 2))
        self.shapes['LOGARITHMIC_UP']   = _norm(np.log1p(x * 10))
        self.shapes['LOGARITHMIC_DOWN'] = _norm(-np.log1p(x * 10))
        step = np.zeros(N); step[N//2:] = 1.0
        self.shapes['STEP_UP']   = step.copy()
        self.shapes['STEP_DOWN'] = 1.0 - step

        # Category 2: Reversals (4 base × 2)
        self.shapes['SYMMETRIC_V_UP']     = _norm(np.abs(x - 0.5))
        self.shapes['SYMMETRIC_V_DOWN']   = _norm(-np.abs(x - 0.5))
        self.shapes['ROUNDED_U_UP']       = _norm((x - 0.5) ** 2)
        self.shapes['ROUNDED_U_DOWN']     = _norm(-((x - 0.5) ** 2))
        self.shapes['FRONT_SKEWED_UP']    = _norm(np.exp(-3 * x) * -1 + 1)
        self.shapes['FRONT_SKEWED_DOWN']  = _norm(np.exp(-3 * x))
        self.shapes['BACK_SKEWED_UP']     = _norm(1 - np.exp(-3 * (1 - x)))
        self.shapes['BACK_SKEWED_DOWN']   = _norm(np.exp(-3 * (1 - x)))

        # Category 3: Volatility (symmetrical, 4)
        self.shapes['SINE_WAVE']          = _norm(np.sin(2 * np.pi * x))
        self.shapes['DAMPED_OSCILLATOR']  = _norm(np.exp(-2 * x) * np.sin(4 * np.pi * x))
        self.shapes['EXPAND_OSCILLATOR']  = _norm(np.exp(x) * np.sin(4 * np.pi * x))
        self.shapes['FLATLINE']           = np.ones(N)

    def classify_trajectory(self, price_segment):
        """Classify a price segment against the 20 seed primitives.

        Args:
            price_segment: array of N raw closing prices (not pre-normalized)

        Returns:
            (best_shape_name, correlation) or ('NOISE', best_corr) if below threshold
        """
        seg = np.asarray(price_segment, dtype=float)
        if len(seg) != self.N:
            return 'NOISE', 0.0

        mn, mx = seg.min(), seg.max()
        if mx - mn < 1e-12:
            return 'FLATLINE', 1.0

        normed = (seg - mn) / (mx - mn)

        best_name = 'NOISE'
        best_corr = -999.0

        for name, template in self.shapes.items():
            if template.std() < 1e-12:
                continue
            r = np.corrcoef(normed, template)[0, 1]
            if np.isnan(r):
                continue
            if r > best_corr:
                best_corr = r
                best_name = name

        if best_corr < self.CORR_THRESHOLD:
            return 'NOISE', best_corr

        return best_name, best_corr

    def save(self, path: str):
        """Serialize library to pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'SeedPrimitiveLibrary':
        """Load library from pickle."""
        with open(path, 'rb') as f:
            return pickle.load(f)
```

### Wire into waveform_standalone.py

Replace the inline class at `tools/waveform_standalone.py:1671` with:
```python
from training.seed_library import SeedPrimitiveLibrary
```

### Generate the library file

After creating the module, run once:
```bash
python -c "from training.seed_library import SeedPrimitiveLibrary; SeedPrimitiveLibrary(16).save('checkpoints/seed_library.pkl'); print('OK')"
```

### Pass/Fail

- `python -c "from training.seed_library import SeedPrimitiveLibrary; lib = SeedPrimitiveLibrary(16); print(lib.classify_trajectory([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]))"` → should print `('LINEAR_UP', 1.0)`
- `python -m py_compile training/seed_library.py` → no errors
- `python -c "from training.seed_library import SeedPrimitiveLibrary; lib = SeedPrimitiveLibrary(16); lib.save('/tmp/test.pkl'); lib2 = SeedPrimitiveLibrary.load('/tmp/test.pkl'); assert lib2.classify_trajectory([16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1])[0] == 'LINEAR_DOWN'; print('PASS')"` → PASS

---

## Part 2: Add 4h Macro Worker to Belief Network

The waveform analysis found that 4h features hold positions #2-#8 in direction prediction
importance. The live system currently tops out at 1h. Add a 4h (14400s) worker.

**Key constraint**: A 4h bar = 14,400 seconds. Resampling one day of 15s bars to 4h gives
only 1-2 bars — not enough for `batch_compute_states` (needs ≥5 bars). The 4h worker must
receive pre-loaded ATLAS data like 5s/1s workers.

**Data**: `DATA/ATLAS/4h/` (12 monthly parquets) and `DATA/ATLAS_OOS/4h/` (2 parquets) exist.

### File: `training/timeframe_belief_network.py`

#### Change 1: Add 4h to TF configuration (lines 297-298)

```python
# BEFORE:
TIMEFRAMES_SECONDS = [3600, 1800, 900, 300, 180, 60, 30, 15, 5, 1]
TF_WEIGHTS         = [4.0,  3.5,  3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.1]

# AFTER:
TIMEFRAMES_SECONDS = [14400, 3600, 1800, 900, 300, 180, 60, 30, 15, 5, 1]
TF_WEIGHTS         = [5.0,   4.0,  3.5,  3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.1]
```

#### Change 2: Add 4h to TF labels (line ~320)

```python
# Add 14400: '4h' to _TF_LABELS dict
_TF_LABELS = {14400:'4h', 3600:'1h', 1800:'30m', 900:'15m', 300:'5m',
              180:'3m',  60:'1m',   30:'30s',   15:'15s',   5:'5s', 1:'1s'}
```

#### Change 3: Add `df_4h` parameter to `prepare_day()` (line 391)

```python
# BEFORE:
def prepare_day(self, df_micro, states_micro=None, df_5s=None, df_1s=None):

# AFTER:
def prepare_day(self, df_micro, states_micro=None, df_5s=None, df_1s=None, df_4h=None):
```

#### Change 4: Handle supra-resolution 4h in the TF loop (line 413-441)

In the `for tf_secs in self.active_timeframes:` loop, add BEFORE the resample block:

```python
            # Supra-resolution TFs (4h): too few bars per day from resampling.
            # Use externally-supplied data like sub-resolution workers.
            if tf_secs == 14400 and df_4h is not None:
                try:
                    _df_4h = df_4h.copy()
                    if not isinstance(_df_4h.index, pd.DatetimeIndex):
                        if 'timestamp' in _df_4h.columns:
                            _df_4h.index = pd.to_datetime(_df_4h['timestamp'], unit='s')
                    states = self.engine.batch_compute_states(_df_4h, use_cuda=True)
                    self.workers[14400].prepare(states)
                    continue
                except Exception as e:
                    logger.warning(f"TBN: 4h state compute failed: {e}")
                    self.workers[14400].prepare([])
                    continue
```

### File: `training/orchestrator.py`

#### Change 5: Load 4h data alongside 5s/1s (line ~623)

```python
_df_5s = _load_fine('5s')
_df_1s = _load_fine('1s')
_df_4h = _load_fine('4h')  # ADD THIS
```

#### Change 6: Pass df_4h to prepare_day (line ~639)

```python
# BEFORE:
belief_network.prepare_day(df_15s, states_micro=_states_15s,
                           df_5s=_df_5s, df_1s=_df_1s)

# AFTER:
belief_network.prepare_day(df_15s, states_micro=_states_15s,
                           df_5s=_df_5s, df_1s=_df_1s, df_4h=_df_4h)
```

Do the same for the fallback call (~line 643).

### Pass/Fail

```bash
python -m training.orchestrator --forward-pass --skip-oos
```

- Check report: `4h=X/file` appears in `WORKER STATES LOADED` line
- Worker state count for 4h should be ~6-7 bars per day × days
- No crashes or regressions in trade count / WR / PnL

---

## Part 3: Shape Classification in Forward Pass Direction Hierarchy

After seed library is saved (Part 1), load it in the forward pass and use shape
classification as a direction signal between Priority 0 (oracle) and Priority 1 (logistic).

### File: `training/orchestrator.py`

#### Change 1: Load seed library at forward pass start (~line 398)

```python
# After library/brain loading, before the day loop
from training.seed_library import SeedPrimitiveLibrary as _SeedLib
_seed_lib_path = os.path.join(self.checkpoint_dir, 'seed_library.pkl')
_seed_library = _SeedLib.load(_seed_lib_path) if os.path.exists(_seed_lib_path) else None
if _seed_library:
    print(f"  Seed library: {len(_seed_library.shapes)} shapes loaded")
```

#### Change 2: Build 15m price buffer per day (~line 596, in day loop)

```python
# After loading df_15s for the day, build 15m resample for shape classification
if _seed_library is not None:
    _tmp_15m = df_15s.copy()
    if not isinstance(_tmp_15m.index, pd.DatetimeIndex):
        _tmp_15m.index = pd.to_datetime(_tmp_15m['timestamp'], unit='s')
    _df_shape_15m = _tmp_15m.resample('900s').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    _shape_15m_closes = _df_shape_15m['close'].values
    _shape_15m_ts = _df_shape_15m.index.astype(np.int64) // 10**9
```

#### Change 3: Insert Priority 0.5 in direction hierarchy (~line 1271)

Between Priority 0 (oracle marker) and Priority 1 (logistic regression):

```python
# Priority 0.5: Seed shape classification (16 bars of 15m closes)
if side is None and _seed_library is not None:
    _ts_mask = _shape_15m_ts <= ts  # ts = current entry timestamp
    _recent_15m = _shape_15m_closes[_ts_mask][-16:]
    if len(_recent_15m) == 16:
        _shape_name, _shape_corr = _seed_library.classify_trajectory(_recent_15m)
        if _shape_name != 'NOISE' and _shape_corr >= 0.75:
            _shape_dir = _SeedLib.DIRECTION_MAP.get(_shape_name)
            if _shape_dir is not None:
                side = _shape_dir
                _dir_source = f'seed_{_shape_name}'
```

#### Change 4: Add shape columns to oracle_trade_log.csv record

In the trade record dict (wherever oracle_trade_records are built), add:
```python
'shape_name': _shape_name if _seed_library else '',
'shape_corr': round(_shape_corr, 3) if _seed_library else 0.0,
```

### Pass/Fail

```bash
python -m training.orchestrator --forward-pass --skip-oos
```

- Check oracle_trade_log.csv: `shape_name` column populated
- Check report or log for `dir_source=seed_*` trades
- Count how many trades got direction from shape vs other sources
- Compare WR of shape-directed trades vs others

---

## Part 4: Train Direction Model from Forward Pass Data

Use the full multi-TF physics fingerprint (11 TFs × 16 features = 176D) to train a
GradientBoosting direction model during the IS forward pass.

### File: `training/timeframe_belief_network.py`

#### Change 1: Add `get_stacked_features()` method

Add to `TimeframeBeliefNetwork` class:

```python
def get_stacked_features(self) -> 'np.ndarray | None':
    """Return (N_TFs, 16) physics matrix from all active workers.

    Returns None if fewer than MIN_ACTIVE_LEVELS workers have data.
    """
    n_tfs = len(self.TIMEFRAMES_SECONDS)
    features = np.zeros((n_tfs, 16))
    has_data = 0
    for idx, tf_secs in enumerate(self.TIMEFRAMES_SECONDS):
        w = self.workers.get(tf_secs)
        if w is None or not w._states or w._last_tf_bar_idx < 0:
            continue
        pos = min(w._last_tf_bar_idx, len(w._states) - 1)
        state_raw = w._states[pos]
        state = state_raw['state'] if isinstance(state_raw, dict) else state_raw
        features[idx, :] = self.state_to_features(state, tf_secs)
        has_data += 1
    if has_data < self.MIN_ACTIVE_LEVELS:
        return None
    return features
```

**Note**: `state_to_features()` already exists at line ~565. Verify it accepts
`(state, tf_secs)` and returns a 16D numpy array.

### File: `training/orchestrator.py`

#### Change 2: Collect direction training data during forward pass

At forward pass initialization (near line 697):
```python
_dir_model_rows = []  # list of {'X': 176D array, 'y': 0/1, 'won': bool|None}
```

At trade ENTRY (in the "FIRE" block):
```python
_stacked = belief_network.get_stacked_features()
if _stacked is not None:
    _dir_model_rows.append({
        'X': _stacked.flatten().copy(),
        'y': 1 if side == 'long' else 0,
        'won': None,  # filled at exit
    })
```

At trade EXIT:
```python
# After computing trade result (win/loss):
if _dir_model_rows and _dir_model_rows[-1]['won'] is None:
    _won = (pnl > 0)
    _dir_model_rows[-1]['won'] = _won
    # Correct the direction label: if we lost, the correct direction was opposite
    if not _won:
        _dir_model_rows[-1]['y'] = 1 - _dir_model_rows[-1]['y']
```

#### Change 3: Train and save model after IS forward pass completes

After the day loop, before the report section:
```python
# Train direction model from IS data
if not oos_mode and len(_dir_model_rows) >= 50:
    _completed = [r for r in _dir_model_rows if r['won'] is not None]
    if len(_completed) >= 50:
        from sklearn.ensemble import GradientBoostingClassifier
        _X_dir = np.array([r['X'] for r in _completed])
        _y_dir = np.array([r['y'] for r in _completed])
        _clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)
        _clf.fit(_X_dir, _y_dir)
        _dm_path = os.path.join(_out_dir, 'direction_model.pkl')
        import pickle as _dm_pkl
        with open(_dm_path, 'wb') as _dmf:
            _dm_pkl.dump({
                'model': _clf,
                'n_features': _X_dir.shape[1],
                'n_samples': len(_X_dir),
                'accuracy': (_clf.predict(_X_dir) == _y_dir).mean(),
            }, _dmf)
        print(f"  Direction model saved: {_dm_path} "
              f"({len(_X_dir)} samples, {_X_dir.shape[1]}D, "
              f"train acc={(_clf.predict(_X_dir) == _y_dir).mean()*100:.1f}%)")
```

#### Change 4: Load and use direction model in OOS pass

At forward pass start, load direction model if it exists:
```python
_dir_model_path = os.path.join(self.checkpoint_dir, 'direction_model.pkl')
_direction_model = None
if os.path.exists(_dir_model_path):
    import pickle as _dm_pkl
    with open(_dir_model_path, 'rb') as _dmf:
        _dm = _dm_pkl.load(_dmf)
    _direction_model = _dm['model']
    print(f"  Direction model: {_dm['n_features']}D, {_dm['n_samples']} samples, "
          f"acc={_dm.get('accuracy', 0)*100:.1f}%")
```

In direction hierarchy, insert Priority 0.3 (before shape, after oracle):
```python
# Priority 0.3: Stacked direction model (176D multi-TF physics)
if side is None and _direction_model is not None:
    _stacked = belief_network.get_stacked_features()
    if _stacked is not None:
        _X_pred = _stacked.flatten().reshape(1, -1)
        _p_long = _direction_model.predict_proba(_X_pred)[0][1]
        if abs(_p_long - 0.5) > 0.15:  # confident
            side = 'long' if _p_long > 0.5 else 'short'
            _dir_source = 'stacked_model'
```

### Pass/Fail

```bash
python -m training.orchestrator --forward-pass
```

- `direction_model.pkl` saved in checkpoints/ (or checkpoints/snowflake/ if PP)
- OOS report shows `stacked_model` as a dir_source
- Compare OOS direction accuracy of stacked_model trades vs baseline
- No regression in overall PnL

---

## Part 5: Live Engine Integration

Load `seed_library.pkl` and `direction_model.pkl` in the live engine. Add two new
direction priorities.

### File: `live/live_engine.py`

#### Change 1: Load in `_load_checkpoints()` (~line 1357)

```python
# Seed shape library
_seed_path = os.path.join(cpdir, 'seed_library.pkl')
self._seed_library = None
if os.path.exists(_seed_path):
    from training.seed_library import SeedPrimitiveLibrary
    self._seed_library = SeedPrimitiveLibrary.load(_seed_path)
    logger.info(f"Seed library loaded: {len(self._seed_library.shapes)} shapes")

# Stacked direction model
_dm_path = os.path.join(cpdir, 'direction_model.pkl')
self._direction_model = None
if os.path.exists(_dm_path):
    import pickle
    with open(_dm_path, 'rb') as f:
        _dm = pickle.load(f)
    self._direction_model = _dm['model']
    logger.info(f"Direction model loaded: {_dm['n_features']}D, "
                f"{_dm['n_samples']} samples")
```

#### Change 2: Add priorities in `_determine_direction()` (~line 1216)

After Priority 0 (live_bias), before Priority 1 (signed_mfe):

```python
# Priority 0.3: Stacked direction model (176D multi-TF physics)
if self._direction_model is not None and self._belief_network is not None:
    _stacked = self._belief_network.get_stacked_features()
    if _stacked is not None:
        _X_pred = _stacked.flatten().reshape(1, -1)
        try:
            _p_long = self._direction_model.predict_proba(_X_pred)[0][1]
            if abs(_p_long - 0.5) > 0.15:
                return ('long' if _p_long > 0.5 else 'short',
                        _p_long, 'stacked_model')
        except Exception:
            pass

# Priority 0.5: Seed shape classification (last 16 fifteen-minute bars)
if self._seed_library is not None:
    _agg_df = self._aggregator.df
    if len(_agg_df) >= 960:  # 16 × 60 fifteen-second bars
        # Sample every 60th close to approximate 15m bars
        _15m_closes = _agg_df['close'].values[-960::60]
        if len(_15m_closes) >= 16:
            _shape_name, _shape_corr = self._seed_library.classify_trajectory(
                _15m_closes[-16:])
            if _shape_name != 'NOISE':
                from training.seed_library import SeedPrimitiveLibrary
                _shape_dir = SeedPrimitiveLibrary.DIRECTION_MAP.get(_shape_name)
                if _shape_dir is not None:
                    _p = 0.7 if _shape_dir == 'long' else 0.3
                    return _shape_dir, _p, f'seed_{_shape_name}'
```

### Pass/Fail

```bash
python -m live.launcher --account Sim101 --yolo
```

- Logs show "Seed library loaded: 20 shapes" and "Direction model loaded: 176D"
- After warmup, direction decisions include `stacked_model` and `seed_*` sources
- No crashes, no regression in trade execution

---

## Architecture Notes

### Direction Hierarchy After Integration

**Orchestrator:**
```
P-1:  Ping-pong live bias (PP mode)
P0:   Oracle marker (IS only)
P0.3: Stacked direction model (176D, |P-0.5|>0.15)   ← NEW
P0.5: Seed shape classification (16 bars 15m)          ← NEW
P1:   Per-cluster logistic regression
P2:   Template aggregate bias
P3:   Live DMI / velocity
      Belief network override (if disagrees, flip)
```

**Live engine:**
```
P0:   Live direction bias (PP refinement)
P0.3: Stacked direction model (176D)                  ← NEW
P0.5: Seed shape classification (16 bars 15m)          ← NEW
P1:   Signed MFE regression
P2:   Balanced direction logistic regression
P3:   Template aggregate bias
P4:   Live DMI / velocity
      Belief network override
```

### Risks

1. **4h worker warmup**: Monthly parquet has ~140 bars → plenty for IS/OOS.
   Live engine: 4h worker inactive until enough bars accumulate (~4 trading days).

2. **Early-day shape gap**: First 4 hours lack 16 fifteen-minute bars.
   Falls through to lower priorities gracefully.

3. **IS model overfit**: OOS auto-chain validates immediately.
   Track `stacked_model` trades separately in OOS report.

4. **Dimension mismatch**: Standalone 192D ≠ live 176D.
   Never transfer models. Always retrain from forward pass data.
