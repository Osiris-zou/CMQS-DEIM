# CMQS-DEIM v1.0.0 — Reproducibility Package for TVC Resubmission

This release corresponds to the revised manuscript **“Curriculum Matching-Aware Query Selection for Efficient End-to-End Object Detection”**, submitted to *The Visual Computer*.

## Main released results

- DEIM-S + CMQS: **49.27 AP** on COCO val2017 (`+0.16 AP` over the local DEIM-S baseline).
- DEIM-L + CMQS: **54.58 AP** on COCO val2017 (`+0.21 AP` over the local DEIM-L baseline).

## Release assets

- `cmqs_deim_s_best.pth`
- `cmqs_deim_s_logs.txt`
- `cmqs_deim_l_best.pth`
- `cmqs_deim_l_logs.txt`

Include the generated `SHA256SUMS.txt` values in this release description after preparing the assets with:

```bash
bash scripts/prepare_release_assets.sh \
  /path/to/deim_s_best.pth /path/to/cmqs_deim_s_logs.txt \
  /path/to/deim_l_best.pth /path/to/cmqs_deim_l_logs.txt
```

## Package contents

- complete CMQS runtime overlay for epoch propagation and matching-aware query selection;
- DEIM-S and DEIM-L baseline/CMQS configurations;
- Tables 4–6 ablation configurations;
- machine-readable manuscript results for Tables 1, 2 and 4–9;
- training, evaluation, profiling, seed-pair and visualization scripts;
- environment, upstream-version and release documentation.

## Citation

The associated manuscript is under review. This v1.0.0 software release is archived on Zenodo under DOI 10.5281/zenodo.20815315.
