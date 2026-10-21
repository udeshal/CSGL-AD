import os
import numpy as np
import pandas as pd

def load_and_stack_npy(directory):
    npy_files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])
    stacked_data = []
    channel_order = []
    feature_dim = None

    for f in npy_files:
        file_path = os.path.join(directory, f)
        data = np.load(file_path)

        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if feature_dim is None:
            feature_dim = data.shape[1]
        elif data.shape[1] != feature_dim:
            print(f"Skipping {f}: feature dimension mismatch ({data.shape[1]} vs {feature_dim})")
            continue

        stacked_data.append(data)
        channel_order.append(os.path.splitext(f)[0])  # channel name (e.g., 'C-1')

        print(f"Loaded {f}: shape = {data.shape}")

    if not stacked_data:
        raise ValueError("No compatible .npy files found.")

    result = np.concatenate(stacked_data, axis=0)
    print(f"\nStacked shape: {result.shape} from {len(channel_order)} channels\n")
    return result, channel_order

def generate_anomaly_labels(channel_order, directory, csv_path):
    label_df = pd.read_csv(csv_path)
    full_labels = []

    for chan in channel_order:
        row = label_df[label_df['chan_id'] == chan]
        test_file = os.path.join(directory, chan + '.npy')
        data = np.load(test_file)
        T = data.shape[0]

        # Default all normal
        labels = np.zeros(T, dtype=int)

        if not row.empty:
            intervals = eval(row.iloc[0]['anomaly_sequences'])  # e.g., [(100, 200), (300, 350)]
            for start, end in intervals:
                labels[start:end+1] = 1

        full_labels.append(labels)

    # Stack all labels along time axis (same as stacked test)
    stacked_labels = np.concatenate(full_labels, axis=0)
    print(f"Anomaly label array shape: {stacked_labels.shape}")
    return stacked_labels
