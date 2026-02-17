import pandas as pd
import os
import numpy as np

# Configuration - Matching the Notebook
BASE_PATH = "/content/drive/MyDrive/1 Skripsi/skrip16feb"
DATASET_FILE = os.path.join(BASE_PATH, "HFV_dataset.csv")
SPLIT_MAP_FILE = os.path.join(BASE_PATH, "alpha_train_test_split_map.csv")

TARGET_VPN_APPS = [
    'VPN_Skype', 'VPN_BitTorrent', 'VPN_Hangout',
    'VPN_Facebook', 'VPN_YouTube', 'VPN_Email'
]

def check_file(path, description):
    print(f"\n{'='*50}")
    print(f"Checking {description}: {path}")
    if not os.path.exists(path):
        print("❌ FILE NOT FOUND!")
        return None
    
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return None

def analyze_dataset():
    print("--- STARTING DATASET VERIFICATION ---")
    
    # 1. Analyze Dataset
    df = check_file(DATASET_FILE, "HFV Dataset")
    if df is None: return

    print("\n--- Column Analysis ---")
    if 'binary_type' in df.columns:
        print(f"Unique 'binary_type' values:\n{df['binary_type'].value_counts()}")
    else:
        print("❌ 'binary_type' column missing!")

    if 'application' in df.columns:
        print(f"\nUnique 'application' values (Top 20):")
        print(df['application'].value_counts().head(20))
        
        print("\nChecking for Target VPN Apps in dataset:")
        found_any = False
        for app in TARGET_VPN_APPS:
            count = df[df['application'] == app].shape[0]
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {app}: {count} samples")
            if count > 0: found_any = True
            
        if not found_any:
            print("\n⚠️ CRITICAL: None of the target VPN apps were found in the 'application' column.")
            print("Did you mean one of these?")
            # Fuzzy match or just print all VPN-like apps
            vpn_apps = [x for x in df['application'].unique() if 'vpn' in str(x).lower()]
            print(vpn_apps)
    else:
        print("❌ 'application' column missing!")

    # 2. Analyze Split Map
    split_map = check_file(SPLIT_MAP_FILE, "Split Map")
    if split_map is None: return

    # Normalize Columns
    split_map.columns = [c.lower().strip() for c in split_map.columns]
    print(f"\nSplit Map Columns (Normalized): {list(split_map.columns)}")
    
    # Identify Filename Column
    map_fname_col = None
    for c in ['filename', 'file', 'pcap']: 
        if c in split_map.columns: map_fname_col = c
    
    if map_fname_col:
        print(f"Identified filename column in split map: '{map_fname_col}'")
    else:
        print("❌ Could not identify filename column in split map (looked for 'filename', 'file', 'pcap')")
        return

    # 3. Analyze Merge
    print("\n--- Merge Analysis ---")
    if 'filename' not in df.columns:
        print("❌ 'filename' column missing in Dataset! Cannot merge.")
        # Try to guess
        print(f"Dataset columns: {list(df.columns)}")
        return

    # Normalize Filenames for comparison (strip whitespace)
    df_files = set(df['filename'].astype(str).str.strip())
    map_files = set(split_map[map_fname_col].astype(str).str.strip())
    
    common = df_files.intersection(map_files)
    print(f"Unique files in Dataset: {len(df_files)}")
    print(f"Unique files in Split Map: {len(map_files)}")
    print(f"Common files (Overlap): {len(common)}")
    
    if len(common) == 0:
        print("❌ CRITICAL: No common filenames found between dataset and split map.")
        print("Sample Dataset Filenames:", list(df_files)[:3])
        print("Sample SplitMap Filenames:", list(map_files)[:3])
        return

    # 4. Simulate Filtering Workflow
    print("\n--- Simulating Notebook Workflow ---")
    
    # Merge
    # Rename map column to 'filename' for merge
    split_map = split_map.rename(columns={map_fname_col: 'filename'})
    
    # Fix case sensitivity in 'split' column (as per previous fix)
    split_col = None
    for c in ['split', 'set', 'partition', 'split_group']:
        if c in split_map.columns: split_col = c
    
    if split_col:
        print(f"Found split column: '{split_col}'")
        split_map['split'] = split_map[split_col].str.lower()
        
        merged = pd.merge(df, split_map[['filename', 'split']], on='filename', how='inner')
        print(f"Merged Shape: {merged.shape}")
        
        # Filter Binary
        vpn_rows = merged[merged['binary_type'] == 'VPN']
        print(f"Rows with binary_type='VPN': {len(vpn_rows)}")
        
        if len(vpn_rows) == 0:
            print("❌ Filtering for 'VPN' resulted in 0 rows. Check 'binary_type' values above.")
        else:
            # Filter Apps
            target_rows = vpn_rows[vpn_rows['application'].isin(TARGET_VPN_APPS)]
            print(f"Rows matching Target Apps: {len(target_rows)}")
            
            if len(target_rows) == 0:
                print("❌ Filtering for specific Apps resulted in 0 rows (within VPN subset).")
                print("Check if the apps exist but maybe binary_type is not 'VPN'?")
                # Check apps in full dataset
                apps_in_full = merged[merged['application'].isin(TARGET_VPN_APPS)]
                print(f"Target apps in FULL merged dataset (ignoring binary_type): {len(apps_in_full)}")
                if len(apps_in_full) > 0:
                    print("Sample of their binary_type:")
                    print(apps_in_full['binary_type'].value_counts())
    else:
        print("❌ Could not identify split column.")

if __name__ == "__main__":
    analyze_dataset()
