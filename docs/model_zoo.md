# Model Zoo and Main Results

## Released CMQS models

| Model | Epochs | AP | AP50 | AP75 | APs | APm | APl | Config | Checkpoint | Logs |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| DEIM-S + CMQS | 132 | 49.27 | 66.3 | 53.3 | 30.6 | 52.6 | 66.0 | [config](../configs/deim_dfine/deim-s-cmqs.yml) | [v1.0.0 asset](https://github.com/Osiris-zou/CMQS-DEIM/releases/download/v1.0.0/cmqs_deim_s_best.pth) | [v1.0.0 asset](https://github.com/Osiris-zou/CMQS-DEIM/releases/download/v1.0.0/cmqs_deim_s_logs.txt) |
| DEIM-L + CMQS | 58 | 54.58 | 72.3 | 59.2 | 38.4 | 58.9 | 71.3 | [config](../configs/deim_dfine/deim-l-cmqs.yml) | [v1.0.0 asset](https://github.com/Osiris-zou/CMQS-DEIM/releases/download/v1.0.0/cmqs_deim_l_best.pth) | [v1.0.0 asset](https://github.com/Osiris-zou/CMQS-DEIM/releases/download/v1.0.0/cmqs_deim_l_logs.txt) |

The model files correspond to the best-recorded validation AP used in the manuscript. Before publishing the release, record the following provenance for each checkpoint:

- source training command and configuration;
- random seed;
- best-recorded epoch;
- full COCO evaluation summary;
- file size and SHA-256 checksum;
- upstream DEIM revision and initialization checkpoint;
- hardware and environment summary.

## Baseline comparison

| Scale | Local DEIM baseline AP | CMQS AP | ΔAP |
|---|---:|---:|---:|
| DEIM-S | 49.11 | 49.27 | +0.16 |
| DEIM-L | 54.37 | 54.58 | +0.21 |

The seed-level experiments remain available in `results/table9_seed_stability.csv` as complementary evidence. The released S- and L-scale checkpoints and logs document the two principal model-scale improvements reported in Tables 1 and 2.
