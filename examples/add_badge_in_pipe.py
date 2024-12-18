import sys
from pathlib import Path

# the filename is the name of temp file created by jupytext, not an original notebook
filename = sys.argv[1]
# the temp filename for "iteratitive_sense_reconstruction.py" is like "iterative_sense_reconstruction-42_5f4kv.py"
basename = Path(filename).stem.rpartition('-')[0]
badge_svg = 'https://colab.research.google.com/assets/colab-badge.svg'
ipynb_link = f'https://colab.research.google.com/github/PTB-MR/mrpro/blob/main/examples/notebooks/{basename}.ipynb'
badge_markdown = f'[![Open In Colab]({badge_svg})]({ipynb_link}) '
badge_pyprocent = f'# %% [markdown]\n# {badge_markdown}\n'
# import_python = '# %%\ntry:\n    import mrpro\n    del mrpro\nexcept ImportError:\n    %pip install mrpro\n'
import_python = "# %%\nimport importlib\n\nif not importlib.util.find_spec('mrpro'):\n    %pip install mrpro\n"
# the temp files of jupytext have the header which looks like:
# ---
# jupyter:
#   jupytext:
# multiple lines...
# ---

# we need to insert the #markdown cell after the header

# insert the badge_pyprocent string after the second occurrence of '# ---'
split_sequence = '# ---\n'
file = Path(filename)
old = file.read_text()
split_text = old.split(split_sequence)

new = split_text[0] + split_sequence + split_text[1] + split_sequence + badge_pyprocent + import_python + split_text[2]
file.write_text(new)
