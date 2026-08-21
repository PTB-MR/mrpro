#!/bin/bash
set -euo pipefail

# pre-install cuda/cpu-specific torch build first
# CUDA_VERSION build arg selects the variant (cu118, cu121, cu124, cu126, or cpu)
python -m pip install --no-cache-dir torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION:-cpu}"

# parse dependencies, excluding torch/torchvision (already installed above,
# with a specific CUDA build — don't let a generic pypi resolve overwrite it)
dependencies=$(python -c "
import sys
try:
    import tomllib as toml_lib
    load = lambda f: toml_lib.load(f)
    mode = 'rb'
except ImportError:
    import toml as toml_lib
    load = lambda f: toml_lib.load(f)
    mode = 'r'

with open('pyproject.toml', mode) as f:
    pyproject = load(f)

all_deps = (
    pyproject['project']['dependencies'] +
    sum(pyproject['project'].get('optional-dependencies', {}).values(), [])
)
excluded = {'torch', 'torchvision'}
filtered = [d for d in all_deps if d.split('[')[0].split('=')[0].split('>')[0].split('<')[0].strip().lower() not in excluded]
print(' '.join(f'\"{dep}\"' for dep in filtered))
")

if [ -z "$dependencies" ]; then
    echo "ERROR: dependency parsing produced an empty list" >&2
    exit 1
fi
echo "Dependencies to install: $dependencies"

read -ra dep_array <<< "$dependencies"
python -m pip install --no-cache-dir "${dep_array[@]}"

rm -rf /root/.cache
