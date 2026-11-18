#!/bin/bash
set -euo pipefail

# pre-install cuda/cpu-specific torch build first
# CUDA_VERSION build arg selects the variant (e.g. cu126, cu128, or cpu)
python -m pip install --no-cache-dir torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION:-cpu}"

# parse dependencies, excluding torch/torchvision (already installed above,
# with a specific CUDA build; don't let a generic pypi resolve overwrite it)
dependencies=$(python -c "
import re
import tomllib

with open('pyproject.toml', 'rb') as f:
    pyproject = tomllib.load(f)

all_deps = (
    pyproject['project']['dependencies'] +
    sum(pyproject['project'].get('optional-dependencies', {}).values(), [])
)
excluded = {'torch', 'torchvision', 'mrpro'}

def name(d):
    return re.match(r'[A-Za-z0-9._-]+', d.strip()).group(0).lower().replace('_', '-')

print('\n'.join(d for d in all_deps if name(d) not in excluded))
")

if [ -z "$dependencies" ]; then
    echo "ERROR: dependency parsing produced an empty list" >&2
    exit 1
fi
echo "Dependencies to install:"
echo "$dependencies"

mapfile -t dep_array <<< "$dependencies"
python -m pip install --no-cache-dir "${dep_array[@]}"

rm -rf /root/.cache
