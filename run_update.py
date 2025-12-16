import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

input_file = 'CBU5201_miniproject_2526.ipynb'
output_file = 'CBU5201_miniproject_2526_v2.ipynb'

print(f"Loading {input_file}...")
with open(input_file) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')

print("Starting execution...")
try:
    ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
    print("Execution complete.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Saved to {output_file}")
    
except Exception as e:
    print(f"Error during execution: {e}")
    # Save partial execution
    with open('error_notebook.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)


