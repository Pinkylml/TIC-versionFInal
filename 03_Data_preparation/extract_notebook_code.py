import json
import sys

nb_path = sys.argv[1]
out_path = sys.argv[2]

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_cells.append(f"# %% [Code Cell]\n{source}\n\n")
            
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(code_cells)
        
    print(f"Extracted {len(code_cells)} code cells to {out_path}")

except Exception as e:
    print(f"Error: {e}")
