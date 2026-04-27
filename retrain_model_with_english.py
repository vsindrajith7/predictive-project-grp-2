"""
Script to retrain the model with 6 languages (including English).
"""
import json
import subprocess
import sys
import os
import re

def extract_code_from_notebook(notebook_path):
    """Extract Python code from notebook, removing IPython magics."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_lines = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Remove IPython magics (!, %, etc.)
            source = re.sub(r'^[!%].*$', '', source, flags=re.MULTILINE)
            # Skip pip installs - dependencies should be pre-installed
            if 'pip install' not in source:
                code_lines.append(source)
    
    return '\n'.join(code_lines)

def run_notebook_code(notebook_name):
    """Convert notebook to Python and execute."""
    print(f"\n{'='*60}")
    print(f"Running: {notebook_name}")
    print(f"{'='*60}\n")
    
    try:
        # Extract Python code from notebook
        code = extract_code_from_notebook(notebook_name)
        
        # Save as Python file
        py_file = notebook_name.replace('.ipynb', '_exec.py')
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"Executing {py_file}...")
        # Execute
        result = subprocess.run([sys.executable, py_file], capture_output=False)
        
        # Clean up
        if os.path.exists(py_file):
            os.remove(py_file)
        
        if result.returncode != 0:
            print(f"ERROR: Failed to execute {notebook_name}")
            return False
        print(f"\nSUCCESS: {notebook_name} completed")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Extract features for 6 languages
    print("STEP 1: Extracting features for 6 languages (including English)...")
    if not run_notebook_code("feature_engineering.ipynb"):
        print("Stopping: Feature extraction failed")
        sys.exit(1)
    
    # Step 2: Train models with 6 classes
    print("\nSTEP 2: Training models with 6 language classes...")
    if not run_notebook_code("ML Pipeline.ipynb"):
        print("Stopping: Model training failed")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("SUCCESS: Model retraining complete!")
    print("New 6-class models created:")
    print("  - rf_model.pkl (Random Forest)")
    print("  - lr_model.pkl (Logistic Regression)")
    print("Classes: Tamil, Telugu, Kannada, Hindi, Malayalam, English")
    print("="*60)
