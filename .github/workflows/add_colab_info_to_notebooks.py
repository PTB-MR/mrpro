import nbformat

# Load the existing notebook
notebook_path = 'examples/direct_reconstruction.ipynb'
with open(notebook_path, 'r') as f:
    notebook = nbformat.read(f, as_version=4)

# Define the new cell to add (a code cell in this case)
new_cell = nbformat.v4.new_code_cell(source="print('Hello, this is a new cell!')")

# Add the new cell to the notebook (e.g., appending to the end)
notebook.cells.append(new_cell)

# Save the modified notebook
with open(notebook_path, 'w') as f:
    nbformat.write(notebook, f)
