# GitHub Release Assets

The GitHub v1.0.0 release contains the following four manuscript-associated files:

| Asset filename | Required content | README link target |
|---|---|---|
| `cmqs_deim_s_best.pth` | DEIM-S + CMQS checkpoint that produces 49.27 AP | `/releases/download/v1.0.0/cmqs_deim_s_best.pth` |
| `cmqs_deim_s_logs.txt` | Verified DEIM-S + CMQS training and evaluation log containing the reported 49.27 AP result | `/releases/download/v1.0.0/cmqs_deim_s_logs.txt` |
| `cmqs_deim_l_best.pth` | DEIM-L + CMQS checkpoint that produces 54.58 AP | `/releases/download/v1.0.0/cmqs_deim_l_best.pth` |
| `cmqs_deim_l_logs.txt` | Verified DEIM-L + CMQS training and evaluation log containing the reported 54.58 AP result | `/releases/download/v1.0.0/cmqs_deim_l_logs.txt` |

The main README and model-zoo documentation link directly to these v1.0.0 Release assets.

## Preparing the same assets for local verification

```bash
bash scripts/prepare_release_assets.sh \
  /path/to/deim_s_best.pth /path/to/cmqs_deim_s_logs.txt \
  /path/to/deim_l_best.pth /path/to/cmqs_deim_l_logs.txt
```

The script creates release_assets/v1.0.0/, copies the two checkpoints and two verified TXT logs, and generates SHA256SUMS.txt.

## Recommended verification for release updates

1. Evaluate each checkpoint with the corresponding configuration.
2. Confirm the exact AP values in the COCO summary.
3. Inspect the text log files and check that they contain no private paths, credentials or unrelated data.
4. Confirm that the configuration snapshot matches the released YAML.
5. Optionally generate and retain SHA-256 values for local integrity verification.
6. The GitHub v1.0.0 release is archived on Zenodo under DOI 10.5281/zenodo.20815315. The all-versions DOI is 10.5281/zenodo.20815314.
