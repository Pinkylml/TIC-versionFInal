import json

nb_path = "Entrenamiento_XGBoost(1).ipynb"
backup_path = "Entrenamiento_XGBoost(1)_backup.ipynb"

print(f"Reading {nb_path}...")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Save backup
print(f"Saving backup to {backup_path}...")
with open(backup_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

fix_count = 0

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            # Fix 1: Hardcoded path
            if '/content/dataset_experimento_B_boosted.csv' in line:
                line = line.replace('/content/dataset_experimento_B_boosted.csv', 'dataset_experimento_B_boosted.csv')
                fix_count += 1
            
            # Fix 2: gpu_hist -> hist + device=cuda (Part 1)
            # Targeting the specific block structure or just simple replace
            if '"tree_method": "gpu_hist",' in line:
                line = line.replace('"tree_method": "gpu_hist",', '"tree_method": "hist",\n            "device": "cuda",')
                fix_count += 1
            
            new_source.append(line)
        cell["source"] = new_source

print(f"Applied {fix_count} fixes.")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Done. Notebook updated.")
