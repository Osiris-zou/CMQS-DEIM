# Public Release Checklist

- [ ] Record the exact upstream DEIM commit or release used.
- [ ] Run `bash scripts/preflight_check.sh`.
- [ ] Export and review the exact training environment with `scripts/export_environment.sh`.
- [ ] Verify DEIM-S uses `query_select_gt_stop_epoch: 24`.
- [ ] Verify DEIM-L uses `query_select_gt_stop_epoch: 10`.
- [ ] Confirm `query_select_cost_mode: sum` in main experiments.
- [ ] Re-evaluate `cmqs_deim_s_best.pth` and confirm 49.27 AP.
- [ ] Re-evaluate `cmqs_deim_l_best.pth` and confirm 54.58 AP.
- [ ] Run `scripts/prepare_release_assets.sh` and inspect both log archives.
- [ ] Publish the generated SHA-256 checksums in the GitHub Release description.
- [ ] Remove local paths, credentials, `.idea`, `.git` internals and unverified files.
- [ ] Create tag `v1.0.0` and a GitHub Release.
- [ ] Upload the four exact assets listed in `release_assets/README.md`.
- [ ] Confirm the README checkpoint and log links work.
- [ ] Archive the final release on Zenodo.
- [ ] Replace every `ZENODO_DOI_TO_BE_ADDED` placeholder.
- [ ] Review upstream licensing before selecting a license for newly written files.
- [ ] Update citation metadata after article publication.
