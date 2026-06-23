# Validation Report

## Automated checks completed

- Python syntax compilation passed for the three runtime files and all tools.
- All main and ablation YAML files parsed successfully.
- `CITATION.cff` parsed successfully as YAML.
- Unit tests confirmed epoch propagation from `det_engine.py` to `deim.py` and the decoder.
- Unit tests confirmed DEIM-L uses `T_exit = 10`, DEIM-S uses `T_exit = 24`, and both main configurations use `query_select_cost_mode: sum`.
- `scripts/preflight_check.sh` passed.
- All shell scripts passed `bash -n` syntax checks.
- `scripts/prepare_release_assets.sh` was tested with dummy checkpoints and log directories; both log archives passed ZIP integrity checks and checksums were generated.
- Relative Markdown links resolve inside the repository.
- No container-local paths or provisional single-author citation strings remain.
- The repository manifest has been regenerated.

## Main result metadata included

- DEIM-S baseline: 49.11 AP.
- DEIM-S + CMQS: 49.27 AP, with preconfigured checkpoint and log release links.
- DEIM-L baseline: 54.37 AP.
- DEIM-L + CMQS: 54.58 AP, with preconfigured checkpoint and log release links.
- Table 9 seed-level data remain available as complementary stability evidence.

## Manual actions still required before public release

1. Record the exact upstream DEIM commit or release.
2. Run `scripts/export_environment.sh` on the actual training machine and review the output.
3. Prepare the four real release assets with `scripts/prepare_release_assets.sh`.
4. Re-evaluate both released checkpoints and confirm 49.27 AP and 54.58 AP.
5. Inspect the log archives for private paths or credentials.
6. Create GitHub tag and Release `v1.0.0`, upload the four exact assets, and publish their SHA-256 checksums.
7. Archive the final release on Zenodo.
8. Replace every `ZENODO_DOI_TO_BE_ADDED` placeholder.
9. Select and document an appropriate license for newly written CMQS-only files after reviewing upstream DEIM terms.
10. Configure the GitHub About description, website and topics using `docs/github_metadata.md`.
