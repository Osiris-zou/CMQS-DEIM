# Public Release Checklist

## Completed

* [x] Document the official upstream DEIM repository.
* [x] Document a compatibility-validated upstream DEIM commit.
* [x] Describe the three modified DEIM runtime files.
* [x] Add Apache License 2.0.
* [x] Add upstream attribution and modification information in `NOTICE`.
* [x] Add third-party software and asset information.
* [x] Verify DEIM-S uses `query_select_gt_stop_epoch: 24`.
* [x] Verify DEIM-L uses `query_select_gt_stop_epoch: 10`.
* [x] Confirm `query_select_cost_mode: sum` in the main configurations.
* [x] Create GitHub tag and Release `v1.0.0`.
* [x] Upload the DEIM-S and DEIM-L CMQS checkpoints.
* [x] Upload the two verified TXT training logs.
* [x] Confirm the README checkpoint and log links resolve to Release assets.
* [x] Archive Release `v1.0.0` on Zenodo.
* [x] Replace the Zenodo DOI placeholders.
* [x] Add software citation metadata.
* [x] Publish the main manuscript result tables in machine-readable form.

## Remaining manual checks

* [ ] Independently re-evaluate `cmqs_deim_s_best.pth` and confirm 49.27 AP, unless already completed.
* [ ] Independently re-evaluate `cmqs_deim_l_best.pth` and confirm 54.58 AP, unless already completed.
* [ ] Confirm that both TXT logs contain no private paths, credentials or unrelated information.
* [ ] Record an exact software-environment freeze for a future rerun when the training environment is available.
* [ ] Optionally publish SHA-256 checksums for all Release assets.
* [ ] Update the manuscript citation metadata after article publication.
