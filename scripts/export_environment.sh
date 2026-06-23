#!/usr/bin/env bash
set -euo pipefail
mkdir -p environment
python -m pip freeze > environment/pip-freeze.txt
if command -v conda >/dev/null 2>&1; then
  conda env export > environment/conda-environment-full.yml
fi
python - <<'PYINFO' > environment/system-info.txt
import platform
print('python:', platform.python_version())
print('platform:', platform.platform())
try:
    import torch
    print('torch:', torch.__version__)
    print('cuda_available:', torch.cuda.is_available())
    print('torch_cuda:', torch.version.cuda)
    print('cudnn:', torch.backends.cudnn.version())
    if torch.cuda.is_available():
        print('gpu_count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'gpu_{i}:', torch.cuda.get_device_name(i))
except Exception as exc:
    print('torch_info_error:', repr(exc))
PYINFO
echo 'Environment records written under environment/'
