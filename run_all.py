# run_all.py
import papermill as pm

# Execute notebooks in order
pm.execute_notebook('notebooks/01_data_cleaning.ipynb', 'outputs/01_result.ipynb')
pm.execute_notebook('notebooks/02_analysis.ipynb', 'outputs/02_result.ipynb')