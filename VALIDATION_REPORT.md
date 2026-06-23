# Validation Report

## Automated checks completed

* Python syntax compilation passed for the three modified runtime files and the repository tools.
* Main and ablation YAML configurations parsed successfully.
* `CITATION.cff` parsed successfully as YAML.
* Unit tests confirmed explicit epoch propagation from `det_engine.py` to `deim.py` and then to the decoder.
* Configuration checks confirmed that DEIM-S uses `T_exit = 24`, DEIM-L uses `T_exit = 10`, and both main configurations use `query_select_cost_mode: sum`.
* `scripts/preflight_check.sh` passed during repository construction.
* Shell scripts passed syntax checks.
* Relative Markdown links were checked within the repository.

## Public release status

* The DEIM-S and DEIM-L CMQS configurations are publicly available.
* The DEIM-S checkpoint and verified TXT training log corresponding to 49.27 AP are available through the GitHub Release.
* The DEIM-L checkpoint and verified TXT training log corresponding to 54.58 AP are available through the GitHub Release.
* Machine-readable results for Tables 1, 2 and 4–9 are included in the `results/` directory.
* GitHub Release `v1.0.0` has been published.
* Release `v1.0.0` has been archived on Zenodo under DOI `10.5281/zenodo.20815315`.
* The all-versions DOI is `10.5281/zenodo.20815314`.
* The repository is distributed under the Apache License, Version 2.0.
* Upstream DEIM attribution and a compatibility-validated upstream commit are documented.

## Remaining manual checks

* Independently re-evaluate the released DEIM-S checkpoint and confirm 49.27 AP if this has not already been completed.
* Independently re-evaluate the released DEIM-L checkpoint and confirm 54.58 AP if this has not already been completed.
* Confirm that the released TXT logs contain no private paths, credentials or unrelated terminal output.
* Record a complete software-environment freeze for future reruns when access to the original or a reproduced training environment is available.
* Optionally publish SHA-256 checksums for the four Release assets.
* Create a synchronized follow-up release after the current `main` branch documentation and licensing updates are finalized.
