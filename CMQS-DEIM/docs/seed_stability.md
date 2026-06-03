# Seed-Level Stability Protocol

The editor requested statistical stability analysis because the AP gain is modest. We recommend paired seed-level experiments.

## Recommended Seeds

```text
42, 3407, 2024
```

## Paired Run Design

For each seed, train both methods under the same training and evaluation settings:

```text
DEIM-L baseline, seed 42
CMQS, seed 42
DEIM-L baseline, seed 3407
CMQS, seed 3407
DEIM-L baseline, seed 2024
CMQS, seed 2024
```

## Reported Metrics

For each run, report:

```text
AP, AP50, AP75, APs, APm, APl, best epoch
```

Then compute the paired gain:

```text
Delta AP = AP(CMQS) - AP(DEIM-L baseline)
```

## Suggested Table

| Seed | DEIM-L AP | CMQS AP | Delta AP | DEIM-L APs | CMQS APs | Delta APs | Baseline best epoch | CMQS best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 |  |  |  |  |  |  |  |  |
| 3407 |  |  |  |  |  |  |  |  |
| 2024 |  |  |  |  |  |  |  |  |
| Mean ± std |  |  |  |  |  |  | - | - |

If full paired baseline runs cannot be completed, report CMQS variance transparently and avoid claiming complete seed-level robustness.
