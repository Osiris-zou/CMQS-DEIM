# Environment Reproducibility

`environment.yml` and `requirements.txt` describe the reference software stack supplied with this repository. They are intentionally conservative because an exact historical package freeze was not present in the source archive used to construct this release.

Before creating an archival GitHub/Zenodo release, export the actual environment used for the reported runs:

```bash
bash scripts/export_environment.sh
```

This creates:

- `environment/conda-environment-full.yml` when Conda is available;
- `environment/pip-freeze.txt`;
- `environment/system-info.txt` with Python, PyTorch, CUDA and cuDNN information.

Commit these generated files only after checking that they do not contain private local paths or credentials.
