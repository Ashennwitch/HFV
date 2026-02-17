import json

nb_path = "HFV_classifier.ipynb"

with open(nb_path, "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = cell["source"]
        for j, line in enumerate(source):
            if "SPLIT_MAP_FILE =" in line:
                print(f"Cell {i}, Line {j}: {repr(line)}")
