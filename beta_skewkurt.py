# --- Delta (δ) Component Extraction Script (OOM Fix / Memory Safe) ---
# FIXES:
# 1. Replaced rdpcap() with PcapReader() to stream packets (avoids RAM spikes).
# 2. Refactored packet processing to a single-pass loop.
# 3. Reduced n_jobs to prevent CPU/RAM saturation.
# 4. UPDATED: Handles multiple input folders.

print("--- Initializing Delta (δ) v2 Component Script (Memory Optimized / Multi-Folder) ---")

try:
    import scapy.all as scapy
except ImportError:
    print("Please run '!pip install scapy' in a Colab cell and restart the runtime.")

import os
import collections
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scapy.all import PcapReader, IP
from scipy.stats import skew, kurtosis

print("All libraries imported successfully.")

# --- PART 1: Configuration & Labeling Map ---

BASE_PATH = "/content/drive/MyDrive/1 Skripsi/"

# UPDATE: List of all input directories
INPUT_DIRS = [
    os.path.join(BASE_PATH, "Dataset/VNAT/flows")
]

OUTPUT_CSV = os.path.join(BASE_PATH, "27jan/VNAT_beta_component.csv")

# --- APP_MAPPING ---
# Updated mapping as requested
APP_MAPPING = {
    'vimeo':       {'Category': 'Streaming', 'App': 'Vimeo'},
    'netflix':     {'Category': 'Streaming', 'App': 'Netflix'},
    'youtube':     {'Category': 'Streaming', 'App': 'YouTube'},
    'voip':        {'Category': 'VoIP',      'App': 'Zoiper'},
    'skype-chat':  {'Category': 'Chat',      'App': 'Skype'},
    'ssh':         {'Category': 'Command & Control', 'App': 'SSH'},
    'rdp':         {'Category': 'Command & Control', 'App': 'RDP'},
    'sftp':        {'Category': 'File Transfer', 'App': 'SFTP'},
    'rsync':       {'Category': 'File Transfer', 'App': 'RSYNC'},
    'scp':         {'Category': 'File Transfer', 'App': 'SCP'}
}

def get_flow_labels(filename):
    """
    Finds the application, category, and binary_type from a filename
    using the new APP_MAPPING and 'nonvpn' logic.
    """
    lower_filename = filename.lower()

    # Updated VPN Logic:
    # If the string "nonvpn" is inside the filename, it is NonVPN.
    # Otherwise, it is VPN.
    if "nonvpn" in lower_filename:
        binary_type = 'NonVPN'
    else:
        binary_type = 'VPN'

    # Updated Loop for new Dictionary Structure
    for keyword, info in APP_MAPPING.items():
        if keyword in lower_filename:
            return info['App'], info['Category'], binary_type

    # If no match found in map, return Unknown
    return "Unknown", "Unknown", binary_type

def calculate_stats(data_list, prefix):
    stats = {}
    stat_names = ['count', 'sum', 'mean', 'std', 'min', 'max', 'median', 'p25', 'p75', 'skew', 'kurt']
    for name in stat_names:
        stats[f"{prefix}_{name}"] = 0.0

    if not data_list:
        return stats

    arr = np.array(data_list)
    stats[f"{prefix}_count"] = float(arr.size)
    stats[f"{prefix}_sum"] = float(np.sum(arr))
    stats[f"{prefix}_mean"] = float(np.mean(arr))
    stats[f"{prefix}_min"] = float(np.min(arr))
    stats[f"{prefix}_max"] = float(np.max(arr))
    stats[f"{prefix}_median"] = float(np.median(arr))
    stats[f"{prefix}_p25"] = float(np.percentile(arr, 25))
    stats[f"{prefix}_p75"] = float(np.percentile(arr, 75))

    if arr.size > 1:
        stats[f"{prefix}_std"] = float(np.std(arr))
        stats[f"{prefix}_skew"] = float(skew(arr))
        stats[f"{prefix}_kurt"] = float(kurtosis(arr))

    return stats

# --- OPTIMIZED FUNCTION ---
def process_pcap_file(filename, base_dir):
    filepath = os.path.join(base_dir, filename)

    application, category, binary_type = get_flow_labels(filename)

    # If get_flow_labels returns None (or if we want to filter Unknowns), handle here.
    # Currently passing "Unknown" through. If you want to skip files not in the map,
    # uncomment the lines below:
    # if application == "Unknown":
    #     return None

    c2s_sizes = []
    c2s_times = []
    s2c_sizes = []
    s2c_times = []

    client_ip = None

    try:
        # MEMORY FIX: Use PcapReader as a context manager for streaming
        with PcapReader(filepath) as pcap_reader:
            for pkt in pcap_reader:
                if IP not in pkt:
                    continue

                # Logic: The first IP we see is the Client (Source)
                if client_ip is None:
                    client_ip = pkt[IP].src

                packet_size = float(pkt[IP].len)
                packet_time = float(pkt.time)

                if pkt[IP].src == client_ip:
                    c2s_sizes.append(packet_size)
                    c2s_times.append(packet_time)
                elif pkt[IP].dst == client_ip:
                    s2c_sizes.append(packet_size)
                    s2c_times.append(packet_time)

    except Exception:
        return None

    if not c2s_times and not s2c_times:
        return None

    # Calculate Flow Duration
    all_times = sorted(c2s_times + s2c_times)
    flow_duration = all_times[-1] - all_times[0] if all_times else 0.0

    # Calculate IATs
    c2s_iats = np.diff(c2s_times).tolist()
    s2c_iats = np.diff(s2c_times).tolist()

    features = {}
    features.update(calculate_stats(c2s_sizes, "c2s_size"))
    features.update(calculate_stats(s2c_sizes, "s2c_size"))
    features.update(calculate_stats(c2s_iats, "c2s_iat"))
    features.update(calculate_stats(s2c_iats, "s2c_iat"))

    features["flow_duration"] = flow_duration
    features["flow_total_packets"] = len(c2s_sizes) + len(s2c_sizes)
    features["flow_total_volume"] = sum(c2s_sizes) + sum(s2c_sizes)

    labels = {
        'filename': filename,
        'application': application,
        'category': category,
        'binary_type': binary_type
    }
    labels.update(features)
    return labels

def main():
    print(f"\n--- PART 1: Extracting Delta (δ) Features + Skew/Kurt ---")

    # 1. Collect all valid files from all input directories
    all_tasks = []

    print("Scanning input directories...")
    for directory in INPUT_DIRS:
        if not os.path.isdir(directory):
            print(f"WARNING: Directory not found, skipping: {directory}")
            continue

        files_in_dir = [f for f in os.listdir(directory) if f.endswith('.pcap') or f.endswith('.pcapng')]
        print(f"  -> Found {len(files_in_dir)} pcaps in: {os.path.basename(directory)}")

        # Store tuple of (filename, directory)
        for f in files_in_dir:
            all_tasks.append((f, directory))

    total_files = len(all_tasks)
    if total_files == 0:
        print("FATAL: No .pcap files found in any directory.")
        return

    print(f"\nTotal files to process: {total_files}")
    print("Processing files... (Streaming mode enabled)")
    start_time = time.time()

    # 2. Parallel Processing
    # We unpack the tuple (f, directory) inside the delayed call
    results = Parallel(n_jobs=2, verbose=5)(
        delayed(process_pcap_file)(f, d) for f, d in all_tasks
    )

    end_time = time.time()
    print(f"File processing finished in {end_time - start_time:.2f} seconds.")

    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("FATAL: No valid data extracted.")
        return

    print(f"Successfully processed {len(valid_results)} files.")
    df_final = pd.DataFrame(valid_results)

    print("\n--- PART 2: Saving Final Dataset ---")
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    try:
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved to: {OUTPUT_CSV}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    if not os.path.exists("/content/drive/MyDrive"):
        print("Please mount your Google Drive first!")
    else:
        main()