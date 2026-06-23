# Environment Reproducibility

The exact historical package freeze used for the reported experiments was not preserved. The provided `environment.yml` and `requirements.txt` therefore describe a compatible reference software stack rather than an exact frozen environment.

For future reruns and software releases, the complete environment should be exported from the training machine and reviewed to remove private paths or credentials before publication.

Before creating an archival GitHub/Zenodo release, export the actual environment used for the reported runs:

```bash
bash scripts/export_environment.sh
```

This creates:

- `environment/conda-environment-full.yml` when Conda is available;
- `environment/pip-freeze.txt`;
- `environment/system-info.txt` with Python, PyTorch, CUDA and cuDNN information.

Commit these generated files only after checking that they do not contain private local paths or credentials.
