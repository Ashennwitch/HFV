import json
import os

nb_path = "HFV_classifier.ipynb"

with open(nb_path, "r") as f:
    nb = json.load(f)

# Re-inject source code with the fix
new_source_code_str = r"""# --- PESV v3 "Championship" with Hyperparameter Tuning (Hierarchical Flow) ---
#
# Workflow:
# 1. Binary Classification (VPN vs Non-VPN) on FULL dataset.
# 2. Filter for VPN traffic only.
# 3. Filter for specific target applications.
# 4. Classify Category and Application within the VPN subset.

print("--- Initializing HFV Hierarchical Classification ---")

import pandas as pd
import numpy as np
import time
import os
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("WARNING: XGBoost not installed. Skipping XGBoost.")
    HAS_XGB = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- PART 1: Configuration ---

BASE_PATH = "/content/drive/MyDrive/1 Skripsi/skrip16feb"
FINAL_PESV_FILE = os.path.join(BASE_PATH, "HFV_dataset.csv")
SPLIT_MAP_FILE = os.path.join(BASE_PATH, "alpha_train_test_split_map.csv")

# Target Applications for the VPN Step
# UPDATED: Removed 'VPN_' prefix to match dataset labels found in 'application' column.
TARGET_VPN_APPS = [
    'Skype', 'BitTorrent', 'Hangout',
    'Facebook', 'YouTube', 'Email'
]

TEST_SET_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3
N_ITER_SEARCH = 10

# --- PART 2: Model & Parameter Definitions ---

MODEL_CONFIGS = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__criterion": ["gini", "entropy"]
        }
    }
}

if HAS_XGB:
    MODEL_CONFIGS["XGBoost"] = {
        "model": XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'),
        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [3, 6, 10],
            "classifier__subsample": [0.8, 1.0]
        }
    }

# --- PART 3: Data Loading Helper ---

def load_data_and_features():
    print(f"\n--- Loading Full Dataset from {FINAL_PESV_FILE} ---")
    if not os.path.exists(FINAL_PESV_FILE):
        print(f"FATAL ERROR: Could not find dataset at '{FINAL_PESV_FILE}'")
        return None, None

    df = pd.read_csv(FINAL_PESV_FILE)
    
    # Define Feature Columns
    all_cols = set(df.columns)
    alpha_cols = sorted([c for c in all_cols if c.startswith('alpha_pp_')])
    delta_cols = sorted([c for c in all_cols if c.startswith(('c2s_', 's2c_', 'flow_', 'total_'))])
    gamma_cols = sorted([c for c in all_cols if c.startswith('burst_')])

    feature_sets = {
        "Alpha'' (α'') only": alpha_cols,
        "Delta (δ) only": delta_cols,
        "Gamma' (γ') only": gamma_cols,
        "Alpha'' + Delta": alpha_cols + delta_cols,
        "Alpha'' + Gamma'": alpha_cols + gamma_cols,
        "Delta + Gamma'": delta_cols + gamma_cols,
        "Full (α'' + δ + γ')": alpha_cols + delta_cols + gamma_cols,
    }

    return df, feature_sets

# --- PART 4: Tuned Classification Task ---

def run_tuned_classification(df_train, df_test, target_label, feature_set_name, feature_cols, model_name, config):
    print(f" > Training {model_name} on {feature_set_name} ({len(feature_cols)} feats)...")

    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = df_train[target_label]
    X_test = df_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = df_test[target_label]

    # Encode labels (Union of train/test to handle missing classes in splits)
    le = LabelEncoder()
    all_labels = pd.concat([y_train, y_test], axis=0)
    le.fit(all_labels)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    class_names = [str(c) for c in le.classes_]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', config["model"])
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=config["params"],
        n_iter=N_ITER_SEARCH,
        scoring='f1_macro',
        cv=CV_FOLDS,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )

    start_time = time.time()
    try:
        search.fit(X_train, y_train_encoded)
        best_model = search.best_estimator_
        best_params = search.best_params_
        y_pred = best_model.predict(X_test)

        report = classification_report(y_test_encoded, y_pred, target_names=class_names, output_dict=True)
        
        return {
            'model': model_name,
            'feature_set': feature_set_name,
            'accuracy': report['accuracy'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'f1_macro': report['macro avg']['f1-score'],
            'time': time.time() - start_time,
            'best_params': str(best_params).replace("classifier__", "")
        }
    except Exception as e:
        print(f"    ERROR in training: {e}")
        return None

# --- PART 5: Main Orchestration ---

def print_results(task_name, metrics_list):
    print(f"\n{'='*100}")
    print(f"--- FINAL RESULTS: {task_name} ---")
    print(f"{'='*100}\n")
    
    sorted_res = sorted([m for m in metrics_list if m], key=lambda x: x['f1_macro'], reverse=True)
    
    print(f"{'Model':<15} | {'Feature Set':<22} | {'Acc':<6} | {'F1(W)':<6} | {'F1(Mac)':<8} | {'Time':<5}")
    print("-" * 80)
    for r in sorted_res:
        print(f"{r['model']:<15} | {r['feature_set']:<22} | {r['accuracy']:.4f} | {r['f1_weighted']:.4f} | {r['f1_macro']:.4f}   | {r['time']:<5.1f}")
    
    print(f"\n--- 🏆 Top 3 Models for {task_name} ---")
    for i, r in enumerate(sorted_res[:3]):
        print(f"{i+1}. {r['model']} [{r['feature_set']}] - Acc: {r['accuracy']:.4f}, Macro F1: {r['f1_macro']:.4f}")
        print(f"   Params: {r['best_params']}")

def main():
    # 1. Load Data
    df, feature_sets = load_data_and_features()
    if df is None: return

    # 2. Merge with Split Map (Global Split)
    if os.path.exists(SPLIT_MAP_FILE):
        print(f"\n--- Loading Split Map: {SPLIT_MAP_FILE} ---")
        split_map = pd.read_csv(SPLIT_MAP_FILE)
        # Normalization (Fix for Case Sensitivity)
        split_map.columns = [c.lower().strip() for c in split_map.columns]
        
        col_map = {'filename': None, 'split': None}
        for c in split_map.columns:
            if c in ['filename', 'file', 'pcap']: col_map['filename'] = c
            if c in ['split', 'set', 'partition', 'split_group']: col_map['split'] = c
            
        if not col_map['filename'] or not col_map['split']:
            print("ERROR: Invalid Split Map Columns")
            return
            
        split_map = split_map.rename(columns={col_map['filename']: 'filename', col_map['split']: 'split'})
        split_map['split'] = split_map['split'].str.lower() # Vital Fix
        
        df_merged = pd.merge(df, split_map[['filename', 'split']], on='filename', how='inner')
        print(f"Data shape after merge: {df_merged.shape}")
    else:
        print("FATAL ERROR: Split map not found. Aborting to prevent leakage.")
        return

    # ---------------------------------------------------------
    # STEP 1: BINARY CLASSIFICATION (VPN vs NonVPN)
    # ---------------------------------------------------------
    print(f"\n{'#'*40}")
    print(" STEP 1: BINARY CLASSIFICATION (VPN vs NonVPN) ")
    print(f"{'#'*40}")
    
    df_train = df_merged[df_merged['split'] == 'train']
    df_test = df_merged[df_merged['split'] == 'test']
    print(f"Binary Train Samples: {len(df_train)}, Test Samples: {len(df_test)}")
    
    binary_metrics = []
    for model_name, config in MODEL_CONFIGS.items():
        for fs_name, fs_cols in feature_sets.items():
            if fs_cols:
                m = run_tuned_classification(df_train, df_test, 'binary_type', fs_name, fs_cols, model_name, config)
                if m: binary_metrics.append(m)
    
    print_results("Binary Type", binary_metrics)

    # ---------------------------------------------------------
    # STEP 2: VPN-SPECIFIC CLASSIFICATION (Category & App)
    # ---------------------------------------------------------
    print(f"\n{'#'*40}")
    print(" STEP 2: VPN-SPECIFIC CLASSIFICATION ")
    print(f"{'#'*40}")
    
    # Filter: Keep only VPN rows
    df_vpn = df_merged[df_merged['binary_type'] == 'VPN'].copy()
    
    # Filter: Keep only Target Apps
    print(f"Filtering for specific apps: {TARGET_VPN_APPS}")
    df_vpn_filtered = df_vpn[df_vpn['application'].isin(TARGET_VPN_APPS)]
    
    df_train_vpn = df_vpn_filtered[df_vpn_filtered['split'] == 'train']
    df_test_vpn = df_vpn_filtered[df_vpn_filtered['split'] == 'test']
    
    print(f"VPN Filtered Data - Total: {len(df_vpn_filtered)}")
    print(f"VPN Train Samples: {len(df_train_vpn)}, Test Samples: {len(df_test_vpn)}")
    
    if len(df_train_vpn) == 0 or len(df_test_vpn) == 0:
        print("WARNING: No samples left after filtering for specific VPN apps. Check dataset labels.")
        return

    # Run Tasks
    for task in ['category', 'application']:
        print(f"\n--- Target: {task} (Within VPN) ---")
        task_metrics = []
        for model_name, config in MODEL_CONFIGS.items():
            for fs_name, fs_cols in feature_sets.items():
                if fs_cols:
                    m = run_tuned_classification(df_train_vpn, df_test_vpn, task, fs_name, fs_cols, model_name, config)
                    if m: task_metrics.append(m)
        
        print_results(f"VPN {task}", task_metrics)

if __name__ == "__main__":
    main()
"""

# Split into lines safely
new_source_lines = [line + "\n" for line in new_source_code_str.splitlines()]
if new_source_lines:
    new_source_lines[-1] = new_source_lines[-1].rstrip("\n")

# Find main cell again
big_cells = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "def main():" in "".join(cell["source"]):
        big_cells.append(i)

if len(big_cells) > 0:
    nb["cells"][big_cells[0]]["source"] = new_source_lines
    # Delete duplicates
    for idx in sorted(big_cells[1:], reverse=True):
        del nb["cells"][idx]

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with corrected App Labels.")
else:
    print("Could not find the main code cell to replace.")
