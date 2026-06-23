# Upstream DEIM Version

CMQS-DEIM is developed as a modification and reproducibility overlay of the official DEIM project.

## Upstream repository

* Project: DEIM
* Repository: `Intellindust-AI-Lab/DEIM`
* Branch: `main`
* Compatibility-validated commit: `bc11dfefc08d79756508c7f8b56c29feb909a4f0`
* Commit date: July 21, 2025
* Upstream license: Apache License 2.0

The exact historical upstream commit used when the original experimental environment was created was not preserved. To provide a fixed and inspectable reference point, the released CMQS-DEIM implementation has been checked against the DEIM repository snapshot identified above.

## Modified core files

The following DEIM files were modified for CMQS-DEIM:

* `engine/deim/dfine_decoder.py`
* `engine/deim/deim.py`
* `engine/solver/det_engine.py`

The main modifications include:

* explicit propagation of the current training epoch;
* curriculum matching-aware candidate-query scoring;
* per-image score normalization;
* ground-truth matching-cost scheduling;
* top-K decoder query selection;
* optional query-analysis export.

Applicable upstream copyright, license, and attribution notices are retained. Additional information is provided in `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, and `docs/changelog_from_deim.md`.
