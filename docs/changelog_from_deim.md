# Changes Relative to Upstream DEIM

## Core runtime files

### `engine/deim/dfine_decoder.py`

- adds classification, geometric-stability and minimal matching-cost ranking terms;
- performs per-image z-score normalization;
- applies $\beta(t)$ and the configured exit epoch;
- disables GT cost at/after the exit epoch and during inference;
- defaults to fixed cost weights through `query_select_cost_mode: sum`;
- uses the configured denoising `box_noise_scale`;
- fails loudly when CMQS training is started without a current epoch;
- provides an optional selected-query export for offline analysis.

### `engine/deim/deim.py`

Accepts `epoch` in `forward` and forwards it to the decoder.

### `engine/solver/det_engine.py`

Passes `epoch=epoch` in both AMP and non-AMP training branches.

The optional analysis export is disabled by default and does not change selected indices.
