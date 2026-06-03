# Changelog from the Upstream DEIM Codebase

This document summarizes the CMQS-specific changes relative to the upstream DEIM implementation.

## Core Modified File

### `engine/deim/dfine_decoder.py`

Main additions:

1. Added CMQS configuration options to the decoder:
   - `query_select_method`
   - `query_select_alpha`
   - `query_select_beta`
   - `query_select_gamma`
   - `query_select_use_gt`
   - `query_select_gt_stop_epoch`
   - `query_select_cost_mode`
   - stage-wise cost weights for classification, L1 and GIoU terms.

2. Added curriculum matching-aware query scoring:
   - classification confidence term;
   - anchor/proposal-based geometric stability term;
   - minimal matching cost term during early training.

3. Added per-image query-to-GT cost construction and per-image minimal-cost extraction.

4. Added per-image score normalization before score fusion.

5. Modified decoder query selection so that `query_select_method: cost_aware` ranks candidate queries by CMQS score instead of classification-only confidence.

6. GT-cost guidance is disabled when `epoch >= query_select_gt_stop_epoch` and is never used during inference.

## Configuration Files

### `configs/deim_dfine/deim-l-cmqs.yml`

Main DEIM-L CMQS configuration used in the paper.

### `configs/deim_dfine/deim-l-baseline.yml`

Local DEIM-L baseline configuration for fair comparison.

### `configs/deim_dfine/deim-s-cmqs.yml`

Additional DEIM-S CMQS configuration for model-scale validation.

## Analysis Tools

The `tools/` directory contains scripts for:

- computational profiling;
- parameter and FLOP calculation;
- query visualization case selection;
- Figure 6 and Figure 7 visualization generation.

These tools are not required for standard training or inference but are included to improve reproducibility of the paper's analyses.
