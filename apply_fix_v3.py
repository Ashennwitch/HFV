import json

nb_path = "HFV_classifier.ipynb"

with open(nb_path, "r") as f:
    nb = json.load(f)

# --- Fix Cell 0 (VPN_ONLY) ---
cell0 = nb["cells"][0]
source0 = cell0["source"]
new_source0 = []
c0_fixed_path = False
c0_fixed_norm = False

for line in source0:
    # 1. Path Fix
    if 'SPLIT_MAP_FILE = os.path.join(BASE_PATH, "VPNOnly-train_test_split_map.csv")' in line:
        new_source0.append('    # UPDATED: Use the master split map from alpha2 to prevent leakage\n')
        new_source0.append('    SPLIT_MAP_FILE = os.path.join(BASE_PATH, "train_test_split_map.csv")\n')
        c0_fixed_path = True
    # 2. Normalization Fix
    elif "split_map = split_map.rename(columns={col_map['filename']: 'filename', col_map['split']: 'split'})" in line:
        new_source0.append(line)
        if not c0_fixed_norm: # Prevent double insertion if re-run
             new_source0.append('\n')
             new_source0.append('        # FIX: Normalize split column to lowercase (alpha2 saves as TRAIN/TEST, we need train/test)\n')
             new_source0.append("        split_map['split'] = split_map['split'].str.lower()\n")
             c0_fixed_norm = True
    else:
        new_source0.append(line)

cell0["source"] = new_source0

# --- Fix Cell 1 (FULL) ---
cell1 = nb["cells"][1]
source1 = cell1["source"]
new_source1 = []
c1_fixed_path = False
c1_fixed_norm = False

for line in source1:
    # 1. Path Fix
    if 'SPLIT_MAP_FILE = os.path.join(BASE_PATH, "train_test_split_map.csv")' in line:
        new_source1.append('    # UPDATED: Point to skrip16feb folder where alpha2 saves the map\n')
        new_source1.append('    SPLIT_MAP_FILE = os.path.join(BASE_PATH, "skrip16feb", "train_test_split_map.csv")\n')
        c1_fixed_path = True
    # 2. Normalization Fix
    elif "split_map = split_map.rename(columns={col_map['filename']: 'filename', col_map['split']: 'split'})" in line:
        new_source1.append(line)
        if not c1_fixed_norm:
             new_source1.append('\n')
             new_source1.append('        # FIX: Normalize split column to lowercase (alpha2 saves as TRAIN/TEST, we need train/test)\n')
             new_source1.append("        split_map['split'] = split_map['split'].str.lower()\n")
             c1_fixed_norm = True
    else:
        new_source1.append(line)

cell1["source"] = new_source1

# Save
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Cell 0 Fixed Path: {c0_fixed_path}, Norm: {c0_fixed_norm}")
print(f"Cell 1 Fixed Path: {c1_fixed_path}, Norm: {c1_fixed_norm}")
