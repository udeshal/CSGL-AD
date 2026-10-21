"""
SMD (Server Machine Dataset) Preprocessor
==========================================
Loads train data, test data, and test labels from separate CSV files
(each with a header row) and returns four numpy arrays ready for use
with run_pipeline().

Expected CSV layout
-------------------
  train_csv  : header row + N_train rows × C feature columns (no label column)
  test_csv   : header row + N_test  rows × C feature columns (no label column)
  labels_csv : header row + N_test  rows × 1 column  (binary 0/1 per timestep)

Returns
-------
  train_data   : np.ndarray  shape (N_train, C)  float32
  train_labels : np.ndarray  shape (N_train,)    int   — all zeros (normal)
  test_data    : np.ndarray  shape (N_test,  C)  float32
  test_labels  : np.ndarray  shape (N_test,)     int   — binary {0, 1}

  Returns None if any of the three files is not found.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_smd(
    train_csv: str | Path,
    test_csv:  str | Path,
    labels_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load and validate the SMD dataset from three CSV files.

    Args:
        train_csv:  Path to the training data CSV (header + feature columns).
        test_csv:   Path to the test data CSV    (header + feature columns).
        labels_csv: Path to the test labels CSV  (header + one binary column).

    Returns:
        (train_data, train_labels, test_data, test_labels) on success, or
        None if any of the three files is not found (a warning is printed for
        each missing file and the function returns immediately without raising).

        train_data   : (N_train, C) float32
        train_labels : (N_train,)   int — all zeros
        test_data    : (N_test,  C) float32
        test_labels  : (N_test,)    int — binary {0, 1}

    Raises:
        ValueError : if test data and test labels have different lengths,
                     or if train/test channel counts differ.
    """
    train_csv  = Path(train_csv)
    test_csv   = Path(test_csv)
    labels_csv = Path(labels_csv)

    # ------------------------------------------------------------------ #
    # 1. Check all files exist — skip gracefully if any are missing
    # ------------------------------------------------------------------ #
    missing = [p for p in (train_csv, test_csv, labels_csv) if not p.exists()]
    if missing:
        for p in missing:
            print(f"  [SKIP] File not found: {p}")
        return None

    # ------------------------------------------------------------------ #
    # 2. Load CSVs — header=0 skips the first row automatically
    # ------------------------------------------------------------------ #
    train_df  = pd.read_csv(train_csv,  header=0)
    test_df   = pd.read_csv(test_csv,   header=0)
    labels_df = pd.read_csv(labels_csv, header=0)

    # ------------------------------------------------------------------ #
    # 3. Convert to numpy
    # ------------------------------------------------------------------ #
    train_data = train_df.values.astype(np.float32)   # (N_train, C)
    test_data  = test_df.values.astype(np.float32)    # (N_test,  C)

    # Labels file: flatten to 1-D regardless of column count
    test_labels = labels_df.values.flatten().astype(int)   # (N_test,)

    # Train labels: all zeros, length = number of training timesteps
    train_labels = np.zeros(train_data.shape[0], dtype=int)   # (N_train,)

    # ------------------------------------------------------------------ #
    # 4. Sanity checks
    # ------------------------------------------------------------------ #
    if test_data.shape[0] != test_labels.shape[0]:
        raise ValueError(
            f"Row count mismatch: test_data has {test_data.shape[0]} rows "
            f"but test_labels has {test_labels.shape[0]} rows."
        )

    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError(
            f"Channel count mismatch: train_data has {train_data.shape[1]} columns "
            f"but test_data has {test_data.shape[1]} columns."
        )

    unique_labels = np.unique(test_labels)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            f"test_labels contains values other than 0 and 1: {unique_labels}"
        )

    # ------------------------------------------------------------------ #
    # 5. Summary
    # ------------------------------------------------------------------ #
    C              = train_data.shape[1]
    anomaly_rate   = test_labels.mean()
    n_anom_windows = test_labels.sum()

    print("SMD dataset loaded successfully")
    print(f"  Channels (C)    : {C}")
    print(f"  Train shape     : {train_data.shape}   "
          f"anomaly_rate={train_labels.mean():.2%}  (all normal by design)")
    print(f"  Test  shape     : {test_data.shape}   "
          f"anomaly_rate={anomaly_rate:.2%}  ({n_anom_windows} anomalous timesteps)")
    print(f"  Test labels     : {test_labels.shape}  "
          f"unique values={list(unique_labels)}")

    return train_data, train_labels, test_data, test_labels


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace these paths with your actual file locations
    TRAIN_CSV  = "data/smd_train.csv"
    TEST_CSV   = "data/smd_test.csv"
    LABELS_CSV = "data/smd_test_labels.csv"

    result = load_smd(
        train_csv  = TRAIN_CSV,
        test_csv   = TEST_CSV,
        labels_csv = LABELS_CSV,
    )

    if result is None:
        print("One or more files were missing — skipping this entity.")
    else:
        train_data, train_labels, test_data, test_labels = result

        # These arrays plug directly into run_pipeline() — you will need to
        # create a val split manually since SMD only ships train/test.
        # A simple approach: use the last 20% of train_data as validation.
        val_split    = int(len(train_data) * 0.8)

        val_data     = train_data[val_split:]
        val_labels   = train_labels[val_split:]   # all zeros (normal)
        train_data   = train_data[:val_split]
        train_labels = train_labels[:val_split]

        print("\nAfter val split:")
        print(f"  train_data   : {train_data.shape}")
        print(f"  train_labels : {train_labels.shape}")
        print(f"  val_data     : {val_data.shape}")
        print(f"  val_labels   : {val_labels.shape}")
        print(f"  test_data    : {test_data.shape}")
        print(f"  test_labels  : {test_labels.shape}")

        # Then call:
        # from temporal_scale_gnn_v3 import run_pipeline, ScaleGNNConfig
        # model, threshold, metrics = run_pipeline(
        #     train_data, train_labels,
        #     val_data,   val_labels,
        #     test_data,  test_labels,
        #     dataset_name="SMD",
        # )