import os
import collections
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# --- SCAPY IMPORT & CHECK ---
try:
    # We import PcapReader instead of rdpcap for memory efficiency
    from scapy.all import PcapReader, IP
except ImportError:
    print("Scapy not found. Installing now...")
    os.system('pip install scapy')
    from scapy.all import PcapReader, IP

print("--- Initializing Gamma-Prime (γ') v3 (Streaming Optimized) ---")

# --- CONFIGURATION ---
BASE_PATH = "/content/drive/MyDrive/1 Skripsi/"

# UPDATE: List of all input directories
INPUT_DIRS = [
    os.path.join(BASE_PATH, "Dataset/VNAT/flows")
]

OUTPUT_CSV = os.path.join(BASE_PATH, "27jan/VNAT_gamma_component.csv")
BURST_IDLE_THRESHOLD = 1.0

# --- APP_MAPPING ---
APP_MAPPING = {
    'vimeo':       {'Category': 'Streaming', 'App': 'Vimeo'},
    'netflix':     {'Category': 'Streaming', 'App': 'Netflix'},
    'youtube':     {'Category': 'Streaming', 'App': 'YouTube'},
    'voip':        {'Category': 'VoIP',       'App': 'Zoiper'},
    'skype-chat':  {'Category': 'Chat',       'App': 'Skype'},
    'ssh':         {'Category': 'Command & Control', 'App': 'SSH'},
    'rdp':         {'Category': 'Command & Control', 'App': 'RDP'},
    'sftp':        {'Category': 'File Transfer', 'App': 'SFTP'},
    'rsync':       {'Category': 'File Transfer', 'App': 'RSYNC'},
    'scp':         {'Category': 'File Transfer', 'App': 'SCP'}
}

# --- HELPER FUNCTIONS ---

def get_flow_labels(filename):
    """
    Parses a filename to get its labels using the APP_MAPPING and onvpn logic.
    """
    lower_filename = filename.lower()

    # VPN Logic
    if "nonvpn" in lower_filename:
        binary_type = 'NonVPN'
    else:
        binary_type = 'VPN'

    # App/Category Logic
    for keyword, info in APP_MAPPING.items():
        if keyword in lower_filename:
            return info['App'], info['Category'], binary_type

    # Fallback
    return "Unknown", "Unknown", binary_type

def calculate_stats(data_list, prefix):
    stats = {}
    stat_names = ['count', 'sum', 'mean', 'std', 'min', 'max', 'median', 'p25', 'p75']
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

    return stats

def get_burst_features(packet_list, prefix):
    """
    Helper function to calculate burst stats for a specific list of packets.
    packet_list: list of (time, size) tuples
    """
    if not packet_list:
        # Return empty stats with correct keys
        empty_feats = {}
        empty_feats[f"{prefix}_total_bursts"] = 0.0
        empty_feats.update(calculate_stats([], f"{prefix}_burst_pkts"))
        empty_feats.update(calculate_stats([], f"{prefix}_burst_vol"))
        empty_feats.update(calculate_stats([], f"{prefix}_burst_dur"))
        empty_feats.update(calculate_stats([], f"{prefix}_burst_idle"))
        return empty_feats

    # Sort by time
    packet_list.sort(key=lambda x: x[0])

    burst_packet_counts = []
    burst_volumes = []
    burst_durations = []
    burst_idle_times = []

    # Init first burst
    current_burst_packets = 1
    current_burst_volume = packet_list[0][1]
    current_burst_start_time = packet_list[0][0]
    last_packet_time = packet_list[0][0]

    for (pkt_time, pkt_size) in packet_list[1:]:
        idle_time = pkt_time - last_packet_time

        if idle_time < BURST_IDLE_THRESHOLD:
            current_burst_packets += 1
            current_burst_volume += pkt_size
        else:
            # End current burst
            burst_duration = last_packet_time - current_burst_start_time
            burst_packet_counts.append(current_burst_packets)
            burst_volumes.append(current_burst_volume)
            burst_durations.append(burst_duration)
            burst_idle_times.append(idle_time)

            # Start new burst
            current_burst_packets = 1
            current_burst_volume = pkt_size
            current_burst_start_time = pkt_time

        last_packet_time = pkt_time

    # Final burst
    burst_duration = last_packet_time - current_burst_start_time
    burst_packet_counts.append(current_burst_packets)
    burst_volumes.append(current_burst_volume)
    burst_durations.append(burst_duration)

    # Compile Features
    features = {}
    features[f"{prefix}_total_bursts"] = float(len(burst_packet_counts))

    features.update(calculate_stats(burst_packet_counts, f"{prefix}_burst_pkts"))
    features.update(calculate_stats(burst_volumes, f"{prefix}_burst_vol"))
    features.update(calculate_stats(burst_durations, f"{prefix}_burst_dur"))
    features.update(calculate_stats(burst_idle_times, f"{prefix}_burst_idle"))

    return features

def process_pcap_file(filename, directory):
    """
    Optimized: Uses PcapReader to stream packets instead of loading whole file to RAM.
    """
    filepath = os.path.join(directory, filename)

    # Labeling
    application, category, binary_type = get_flow_labels(filename)

    c2s_packets = []
    s2c_packets = []
    client_ip = None

    try:
        # PcapReader creates a generator, reading one packet at a time
        with PcapReader(filepath) as packets:
            for pkt in packets:
                if IP in pkt:
                    # Heuristic: The first IP encountered is treated as Client
                    if client_ip is None:
                        client_ip = pkt[IP].src

                    # Extract necessary data immediately (Time, Size)
                    packet_size = float(pkt[IP].len)
                    packet_time = float(pkt.time)

                    # Determine Direction
                    if pkt[IP].src == client_ip:
                        c2s_packets.append((packet_time, packet_size))
                    elif pkt[IP].dst == client_ip:
                        s2c_packets.append((packet_time, packet_size))

    except Exception as e:
        # print(f"Error processing {filename}: {e}") # Uncomment for debug
        return None

    if not c2s_packets and not s2c_packets:
        return None

    # Calculate Directional Features (Same logic as original)
    features = {}
    features.update(get_burst_features(c2s_packets, "c2s"))
    features.update(get_burst_features(s2c_packets, "s2c"))

    # Add Labels
    features['filename'] = filename
    features['application'] = application
    features['category'] = category
    features['binary_type'] = binary_type

    return features

# --- MAIN EXECUTION ---

def main():
    # 1. Collect all valid files
    all_tasks = []

    print("Scanning input directories...")
    for directory in INPUT_DIRS:
        if not os.path.isdir(directory):
            print(f"WARNING: Directory not found, skipping: {directory}")
            continue

        files_in_dir = [f for f in os.listdir(directory) if f.endswith('.pcap')]
        print(f"  -> Found {len(files_in_dir)} pcaps in: {os.path.basename(directory)}")

        for f in files_in_dir:
            all_tasks.append((f, directory))

    total_files = len(all_tasks)
    if total_files == 0:
        print("FATAL: No .pcap files found in any directory.")
        return

    print(f"\nTotal files to process: {total_files}")
    print("Processing in parallel (Optimized for Colab Memory)...")

    # 2. Parallel Processing
    # n_jobs=4 is safer for Colab (prevents OOM kill) compared to -1
    results = Parallel(n_jobs=4, verbose=5)(
        delayed(process_pcap_file)(f, d) for f, d in all_tasks
    )

    valid_results = [r for r in results if r is not None]

    if valid_results:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df = pd.DataFrame(valid_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved Directional Gamma features to {OUTPUT_CSV}")
    else:
        print("No valid results.")

if __name__ == "__main__":
    if not os.path.exists("/content/drive/MyDrive"):
        print("Please mount Google Drive!")
    else:
        main()