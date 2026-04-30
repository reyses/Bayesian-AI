# Auto peak detector calibration - TF=4h

Generated: 2026-04-29 21:31

## Inputs

- Manual peaks: `DATA/regime_seeds\human_peaks_2025-01-02_to_2026-03-20_4h.json` (149 peaks)
- Manual window (bars): 3 -> 891
- Manual window (dates): 2025-01-02 -> 2025-08-13
- Total bars in 4h: 1,743
- Match tolerance: +/- 2 bars

## Calibration sweep

```
 lookback  n_auto  n_manual  n_matched  precision   recall       f1
        3     196       149         91   0.464286 0.610738 0.527536
        5     123       149         65   0.528455 0.436242 0.477941
        8      61       149         37   0.606557 0.248322 0.352381
       10      54       149         32   0.592593 0.214765 0.315271
       15      37       149         21   0.567568 0.140940 0.225806
       20      30       149         18   0.600000 0.120805 0.201117
```

## Selected lookback

- **Best lookback: 3**
- F1: 0.528
- Precision: 0.464  (of auto picks, fraction matching a manual peak)
- Recall: 0.611  (of manual marks, fraction caught by auto)

## Applied to full dataset

- Auto peaks (raw): 382
- After alternation: 313
- Output JSON: `DATA/regime_seeds\auto_peaks_2025-01-02_to_2026-03-20_4h.json`
- Merged file: `DATA/regime_seeds\merged_peaks_2025-01-02_to_2026-03-20_4h.json`

## How to use downstream

```bash
# Re-run the macro segmenter against auto OR merged peaks:
# (Will pick the file with the most peaks via find_peaks_file logic.)
python tools/macro_slope_segmenter.py --tfs 4h
```
