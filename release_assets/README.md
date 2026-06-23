# GitHub Release Assets

Create a GitHub Release with tag `v1.0.0` and upload the following four files using these exact names:

| Asset filename | Required content | README link target |
|---|---|---|
| `cmqs_deim_s_best.pth` | DEIM-S + CMQS checkpoint that produces 49.27 AP | `/releases/download/v1.0.0/cmqs_deim_s_best.pth` |
| `cmqs_deim_s_logs.zip` | Verified DEIM-S + CMQS `log.txt`, `console.log`, config snapshot and evaluation summary | `/releases/download/v1.0.0/cmqs_deim_s_logs.zip` |
| `cmqs_deim_l_best.pth` | DEIM-L + CMQS checkpoint that produces 54.58 AP | `/releases/download/v1.0.0/cmqs_deim_l_best.pth` |
| `cmqs_deim_l_logs.zip` | Verified DEIM-L + CMQS `log.txt`, `console.log`, config snapshot and evaluation summary | `/releases/download/v1.0.0/cmqs_deim_l_logs.zip` |

The links in the main README are already configured for these filenames.

## Prepare assets locally

```bash
bash scripts/prepare_release_assets.sh \
  /path/to/deim_s_best.pth /path/to/deim_s_run_or_log \
  /path/to/deim_l_best.pth /path/to/deim_l_run_or_log
```

The script creates `release_assets/v1.0.0/`, copies the checkpoints, packages the log files or run directories, and generates `SHA256SUMS.txt`.

## Required verification before upload

1. Evaluate each checkpoint with the corresponding configuration.
2. Confirm the exact AP values in the COCO summary.
3. Open the log archives and check that they contain no private paths, credentials or unrelated data.
4. Confirm that the configuration snapshot matches the released YAML.
5. Publish the generated SHA-256 values in the GitHub Release description.
6. After the GitHub Release is final, archive it on Zenodo and replace `ZENODO_DOI_TO_BE_ADDED` in the README and citation metadata.
