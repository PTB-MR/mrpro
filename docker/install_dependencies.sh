# 1. Install torch first, normally, full dependency resolution
python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${1:-cpu}

# 2. Install everything else, eager, but torch excluded from the list so it's never re-resolved
dependencies_no_torch=$(python -c "
import toml
pyproject = toml.load('pyproject.toml')
all_deps = (
    pyproject['project']['dependencies'] +
    sum(pyproject['project'].get('optional-dependencies', {}).values(), [])
)
all_deps = [d for d in all_deps if not d.lower().startswith('torch')]
print(' '.join(f'\"{dep}\"' for dep in all_deps))
")
echo Dependencies to install: $dependencies_no_torch

eval python -m pip install --no-cache-dir --upgrade --upgrade-strategy "eager" $dependencies_no_torch

#clean up
rm -rf /root/.cache
