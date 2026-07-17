# pre-install cpu-version of torch by default
# either use the 1st argument as specifier (cu118, cu124 or cu126)
python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${1:-cpu}

#parse dependencies
python -m pip install --no-cache-dir toml
dependencies=$(python -c "
import toml
pyproject = toml.load('pyproject.toml')
all_deps = (
    pyproject['project']['dependencies'] +
    sum(pyproject['project'].get('optional-dependencies', {}).values(), [])
)
print(' '.join(f'\"{dep}\"' for dep in all_deps))
")
if [ -z "$dependencies" ]; then
    echo "ERROR: dependency parsing produced an empty list" >&2
    exit 1
fi
echo Dependencies to install: $dependencies

# install dependencies
eval python -m pip install --no-cache-dir $dependencies

#clean up
rm -rf /root/.cache
