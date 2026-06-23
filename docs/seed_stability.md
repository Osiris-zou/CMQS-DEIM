# Seed-Level Stability Protocol

The manuscript reports paired DEIM-L experiments for seeds `42`, `3407` and `2024`. For each seed, train the local baseline and CMQS with the same initialization, schedule, optimizer, global batch size, augmentation setup and evaluation protocol.

```bash
bash scripts/run_seed_pair.sh 42
bash scripts/run_seed_pair.sh 3407
bash scripts/run_seed_pair.sh 2024
```

Table 9 reports the validation checkpoint with the highest AP for every run, using the same selection rule for both methods. It does not report final-checkpoint or best-epoch columns.

Historical values used in the manuscript are stored in `results/table9_seed_stability.csv` and `results/table9_seed_summary.csv`. Sample standard deviation is used. With only three seeds, the results are descriptive stability evidence rather than a formal significance test.
