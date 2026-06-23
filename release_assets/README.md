# GitHub Release Assets

Create a GitHub Release with tag `v1.0.0` and upload the following four files using these exact names:

| Asset filename | Required content | README link target |
|---|---|---|
| `cmqs_deim_s_best.pth` | DEIM-S + CMQS checkpoint that produces 49.27 AP | `/releases/download/v1.0.0/cmqs_deim_s_best.pth` |
| `cmqs_deim_s_logs.txt` | Verified DEIM-S + CMQS training and evaluation log containing the reported 49.27 AP result | `/releases/download/v1.0.0/cmqs_deim_s_logs.txt` |
| `cmqs_deim_l_best.pth` | DEIM-L + CMQS checkpoint that produces 54.58 AP | `/releases/download/v1.0.0/cmqs_deim_l_best.pth` |
| `cmqs_deim_l_logs.txt` | Verified DEIM-L + CMQS training and evaluation log containing the reported 54.58 AP result | `/releases/download/v1.0.0/cmqs_deim_l_logs.txt` |

The links in the main README are already configured for these filenames.

## Prepare assets locally

```bash
bash scripts/prepare_release_assets.sh \
  /path/to/deim_s_best.pth /path/to/cmqs_deim_s_logs.txt \
  /path/to/deim_l_best.pth /path/to/cmqs_deim_l_logs.txt
```

The script creates `release_assets/v1.0.0/`, copies the checkpoints, copies the two checkpoints and two verified text logs, and generates `SHA256SUMS.txt`.

## Required verification before upload

1. Evaluate each checkpoint with the corresponding configuration.
2. Confirm the exact AP values in the COCO summary.
3. Inspect the text log files and check that they contain no private paths, credentials or unrelated data.
4. Confirm that the configuration snapshot matches the released YAML.
5. Publish the generated SHA-256 values in the GitHub Release description.
6. The GitHub v1.0.0 release is archived on Zenodo under DOI 10.5281/zenodo.20815315. The all-versions DOI is 10.5281/zenodo.20815314.
