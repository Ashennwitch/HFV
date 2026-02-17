import json
import os

nb_path = "HFV_classifier.ipynb"

if not os.path.exists(nb_path):
    print(f"Error: {nb_path} not found.")
    exit(1)

with open(nb_path, "r") as f:
    nb = json.load(f)

changed = False

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        new_source = []
        cell_changed = False
        
        for line in source:
            # Fix 1: Update VPN_ONLY path
            if 'SPLIT_MAP_FILE = os.path.join(BASE_PATH, "VPNOnly-train_test_split_map.csv")' in line:
                new_source.append('    # UPDATED: Use the master split map from alpha2 to prevent leakage\n')
                new_source.append('    SPLIT_MAP_FILE = os.path.join(BASE_PATH, "train_test_split_map.csv")\n')
                cell_changed = True
                changed = True
            
            # Fix 2: Update FULL path (add skrip16feb)
            # Pattern matches the original line in the "FULL" block which typically points to root
            elif 'SPLIT_MAP_FILE = os.path.join(BASE_PATH, "train_test_split_map.csv")' in line:
                new_source.append('    # UPDATED: Point to skrip16feb folder where alpha2 saves the map\n')
                new_source.append('    SPLIT_MAP_FILE = os.path.join(BASE_PATH, "skrip16feb", "train_test_split_map.csv")\n')
                cell_changed = True
                changed = True

            # Fix 3: Case Sensitivity (Add .str.lower())
            elif "split_map = split_map.rename(columns={col_map['filename']: 'filename', col_map['split']: 'split'})" in line:
                new_source.append(line)
                new_source.append('\n')
                new_source.append('        # FIX: Normalize split column to lowercase (alpha2 saves as TRAIN/TEST, we need train/test)\n')
                new_source.append("        split_map['split'] = split_map['split'].str.lower()\n")
                cell_changed = True
                changed = True
            
            else:
                new_source.append(line)
        
        if cell_changed:
            cell["source"] = new_source

if changed:
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("No changes were made (patterns not found - file might already be fixed).")
