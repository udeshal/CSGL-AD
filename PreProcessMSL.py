import os
import numpy as np
import pandas as pd

# --- Helper to load and trim all channels to same length ---
def load_multivariate_data_trim(directory, channels_to_load):
    data_list = []
    loaded_channels = []
    min_len = float('inf')

    # First pass to find min length among the channels to load
    for chan in channels_to_load:
        f = chan + '.npy'
        filepath = os.path.join(directory, f)
        if os.path.exists(filepath):
            arr = np.load(filepath)
            min_len = min(min_len, len(arr))
            loaded_channels.append(chan) # Only add if file exists

    print(f'Min len in {directory}: {min_len}')

    # Second pass to load and trim data for the channels found
    data_list = []
    # Ensure we iterate through loaded_channels to maintain order
    for chan in loaded_channels:
        f = chan + '.npy'
        filepath = os.path.join(directory, f)
        arr = np.load(filepath)[:min_len]  # trim
        # Ensure the loaded array has the expected 2 dimensions [time, features]
        if arr.ndim == 1:
             # If 1D, reshape to [time, 1] assuming a single feature per file
             arr = arr[:, np.newaxis]
        elif arr.ndim > 2:
             # If more than 2D, take the first 2 dimensions, or raise an error if unexpected
             print(f"Warning: File {filepath} has more than 2 dimensions ({arr.ndim}). Using the first two dimensions.")
             arr = arr[:, :] # This slicing keeps the first two dimensions, assuming they are time and features

        print(f'File trimmed shape for {chan}: {arr.shape}')
        data_list.append(arr)


    print(f'Shape of data_list before concatenating in {directory}: {[d.shape for d in data_list]}')
    # Concatenate along axis 1 to get shape [T, total_features]
    # Add a check to ensure data_list is not empty before concatenating
    if not data_list:
        print(f"Warning: No data loaded from {directory} for the specified channels.")
        return np.array([]), [] # Return empty array and list

    data = np.concatenate(data_list, axis=0)
    print(f'Shape of data after concatenating in {directory}: {data.shape}')
    return data, loaded_channels

# --- Parse interval string like "[(1000, 1200), (1300, 1350)]" ---
def parse_anomaly_intervals(anomaly_str):
    return eval(anomaly_str) if isinstance(anomaly_str, str) else []

# --- Build binary anomaly labelk  vector for test set ---
def generate_label_array(test_len, anomaly_intervals):
    labels = np.zeros(test_len, dtype=int)
    for start, end in anomaly_intervals:
        labels[start:end+1] = 1
    return labels

# --- Main preprocessing pipeline ---
def preprocess_msl_trim(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    csv_path = os.path.join(base_dir, 'labeled_anomalies.csv')

    # Get all channels from train and test directories
    train_files = set([f.replace('.npy', '') for f in os.listdir(train_dir) if f.endswith('.npy')])
    test_files = set([f.replace('.npy', '') for f in os.listdir(test_dir) if f.endswith('.npy')])

    # Find common channels
    common_channels = sorted(list(train_files.intersection(test_files)))
    print(f'Found {len(common_channels)} common channels out of {len(train_files)} train and {len(test_files)} test channels.')

    if not common_channels:
        raise ValueError("No common channels found between train and test directories.")


    # Load train and test data for common channels (trimmed to shortest time length)
    train_data, train_channels = load_multivariate_data_trim(train_dir, common_channels)
    test_data, test_channels = load_multivariate_data_trim(test_dir, common_channels)

    # Double check that the loaded channels are indeed the common channels and in the same order
    if train_channels != common_channels or test_channels != common_channels:
        print("Warning: Loaded channels do not match common channels or order.")
        # Re-align test_data based on common_channels if necessary, though load_multivariate_data_trim should handle this now
        # For robustness, explicitly ensure test_data features align with train_data features based on common_channels order
        # This part might be complex if the original files' feature order is not guaranteed
        # Assuming load_multivariate_data_trim with common_channels handles the order

    # Normalize using train stats - Recalculate mean and std just before use
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    print(f'Shape of mean: {mean.shape}, Shape of std: {std.shape}') # Debug print

    # Add a small epsilon to standard deviation to avoid division by zero
    std = std + 1e-8 # Add epsilon here to avoid modifying the original std array during division

    # Perform normalization
    train_norm = (train_data - mean) / std
    test_norm = (test_data - mean) / std

    # Load labels CSV
    label_df = pd.read_csv(csv_path)

    # Construct anomaly labels for test set
    test_len = test_data.shape[0]
    total_features = len(common_channels) * 25 # Calculate total number of features
    final_labels = np.zeros((test_len, total_features), dtype=int) # Initialize as 2D array

    # Create a mapping from channel name to its index in the common_channels list
    channel_to_index = {chan: i for i, chan in enumerate(common_channels)}

    for chan in common_channels: # Iterate through common channels to ensure alignment
        chan_row = label_df[label_df['chan_id'] == chan]
        if not chan_row.empty:
            anomaly_str = chan_row.iloc[0]['anomaly_sequences']
            intervals = parse_anomaly_intervals(anomaly_str)
            # Find the index of the current channel in the common_channels list
            if chan in channel_to_index:
                chan_index = channel_to_index[chan]
                # Generate labels for the 25 features of this channel
                channel_labels = generate_label_array(test_len, intervals)
                # Expand channel_labels to match the shape of the channel's data (time_steps, 25)
                # Assuming the anomaly applies to all 25 features of a channel
                expanded_channel_labels = np.tile(channel_labels[:, np.newaxis], (1, 25))
                # Get the slice of final_labels corresponding to this channel's features
                start_feature_index = chan_index * 25
                end_feature_index = start_feature_index + 25
                # Update final_labels using maximum to combine anomalies from different channels
                final_labels[:, start_feature_index:end_feature_index] = np.maximum(final_labels[:, start_feature_index:end_feature_index], expanded_channel_labels)

            else:
                print(f"Warning: Channel '{chan}' from common channels not found in channel_to_index mapping.")


    # Combine labels across all features for each time step
    # Anomaly exists at a time step if any feature at that time step is anomalous
    final_labels = np.max(final_labels, axis=1)
    train_len = train_norm.shape[0] # Renamed from len
    train_labels = np.zeros(train_len, dtype=int) # Use train_len


    return train_norm, test_norm, final_labels, train_labels