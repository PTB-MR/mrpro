# pre-install cpu-version of torch to avoid installation of cuda-version via dependencies
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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
echo Dependencies to install: $dependencies

# install dependencies
eval python -m pip install --no-cache-dir --upgrade --upgrade-strategy "eager" $dependencies

#clean up
rm -rf /root/.cache
