import nbformat
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Modify a Jupyter notebook")
parser.add_argument('notebook_path', type=str, help='Path to the Jupyter notebook to modify')

# Parse the arguments
args = parser.parse_args()

# Path to your notebook
notebook_path = args.notebook_path

# Load the existing notebook
with open(notebook_path, 'r') as f:
    notebook = nbformat.read(f, as_version=4)

# Define the new cell to add (a code cell in this case)
new_cell = nbformat.v4.new_code_cell(source="print('Hello, this is a new cell!')")

# Add the new cell to the notebook (e.g., appending to the end)
notebook.cells.append(new_cell)

# Save the modified notebook
with open(notebook_path, 'w') as f:
    nbformat.write(notebook, f)
