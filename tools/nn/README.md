# tools/_nn — Neural Network Training

> Virtual folder. Files in `tools/` root.

| Tool | Purpose |
|---|---|
| [`train_pivot_direction_nn.py`](../train_pivot_direction_nn.py) | CNN on 91D at RM pivots → P(win). 6×15 grid, walk-forward. **Hub** |
| [`train_tier_direction_nn.py`](../train_tier_direction_nn.py) | Same CNN architecture on 9-tier trades. More regularization |
| [`apply_pivot_nn_filter.py`](../apply_pivot_nn_filter.py) | Post-hoc: TAKE/FLIP/SKIP classification by P(win) |
| [`seed_oracle_trainer.py`](../seed_oracle_trainer.py) | Entry classifier from manual seeds. 192D multi-TF features |
| [`shape_primitive_builder.py`](../shape_primitive_builder.py) | UMAP+HDBSCAN on entry+exit primitives |
| [`visual_shape_cnn.py`](../visual_shape_cnn.py) | 2D CNN on candlestick images (level touches) |
| [`calibrate_trajectories.py`](../calibrate_trajectories.py) | Per-TF TrajectoryPredictor calibration |
