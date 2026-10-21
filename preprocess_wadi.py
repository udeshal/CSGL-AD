"""
WADI Dataset Preprocessor
==========================
Reads WADI train and test CSV files and produces four numpy arrays
ready for use with run_pipeline():

    train_data:   (T_train, C)  float32  — normalised sensor readings
    train_labels: (T_train,)    int      — all zeros (WADI train is clean)
    test_data:    (T_test, C)   float32  — normalised sensor readings
    test_labels:  (T_test,)     int      — binary {0=normal, 1=anomaly}

CSV layout assumed (confirmed from WADI format):
    Train file:  col 0 = row number, col 1 = date, col 2 = time, then sensors
    Test file:   same prefix cols, then sensors, last col = anomaly label

Anomaly label encoding in WADI test file:
    -1  → anomaly   (remapped to 1)
     1  → normal    (remapped to 0)

Usage:
    python preprocess_wadi.py
    -- or import and call preprocess_wadi() directly --
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration — update these paths to point to your CSV files
# ---------------------------------------------------------------------------

TRAIN_CSV = Path("WADI_trainingdata.csv")
TEST_CSV  = Path("WADI_attackdata.csv")

# Number of leading columns to drop from both files (row_num, date, time)
COLS_TO_DROP = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_raw(path: Path, has_label_col: bool) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Load one WADI CSV file.

    Returns:
        sensor_df: DataFrame of sensor columns only (floats)
        labels:    Series of raw label values if has_label_col, else None
    """
    print(f"Loading : {path}")
    df = pd.read_csv(path, header=0, low_memory=False)
    print(f"  Raw shape      : {df.shape}")
    print(f"  Columns [0:4]  : {list(df.columns[:4])}")
    print(f"  Columns [-2:]  : {list(df.columns[-2:])}")

    # Drop leading index / timestamp columns
    df = df.iloc[:, COLS_TO_DROP:]
    print(f"  After drop cols: {df.shape}")

    # Separate label column from sensor columns
    labels = None
    if has_label_col:
        labels = df.iloc[:, -1].copy()
        df     = df.iloc[:, :-1]
        print(f"  Label col      : '{labels.name}'  unique={sorted(labels.unique())}")

    return df, labels


def _resolve_nan(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Resolve NaN values in sensor data.

    Strategy (applied in order):
      1. Forward-fill  — propagates last valid reading (sensor held at last value)
      2. Backward-fill — catches NaNs at the very start of the series
      3. Column mean   — fallback for columns that are entirely NaN
    """
    nan_before = df.isna().sum().sum()
    if nan_before == 0:
        print(f"  [{name}] No NaN values found.")
        return df

    print(f"  [{name}] NaN count before: {nan_before}")

    df = df.ffill()   # forward-fill
    df = df.bfill()   # backward-fill for leading NaNs

    # Any column still all-NaN → fill with 0.0 (sensor dead for entire split)
    still_nan = df.isna().sum()
    dead_cols  = still_nan[still_nan > 0].index.tolist()
    if dead_cols:
        print(f"  [{name}] Columns entirely NaN (filled with 0): {dead_cols}")
        df[dead_cols] = 0.0

    nan_after = df.isna().sum().sum()
    print(f"  [{name}] NaN count after : {nan_after}")
    return df


def _binarise_labels(raw: pd.Series) -> np.ndarray:
    """
    Convert WADI raw labels to binary {0, 1}.

    WADI label convention:
       -1 → anomaly → 1
        1 → normal  → 0
    Any other value is treated as normal (0).
    """
    labels = np.zeros(len(raw), dtype=int)
    labels[raw == -1] = 1

    # Defensive: also handle string variants sometimes present in CSV exports
    try:
        str_vals = raw.astype(str).str.strip()
        labels[str_vals == "-1"] = 1
        labels[str_vals == "-1.0"] = 1
    except Exception:
        pass

    return labels


# ---------------------------------------------------------------------------
# Main preprocessor
# ---------------------------------------------------------------------------

def preprocess_wadi(
    train_csv: Path = TRAIN_CSV,
    test_csv:  Path = TEST_CSV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full WADI preprocessing pipeline.

    Args:
        train_csv: path to WADI training CSV file.
        test_csv:  path to WADI test / attack CSV file.

    Returns:
        train_data:   (T_train, C) float32 numpy array
        train_labels: (T_train,)  int numpy array — all zeros
        test_data:    (T_test, C) float32 numpy array
        test_labels:  (T_test,)   int numpy array — binary {0, 1}
    """
    print("=" * 60)
    print("WADI Preprocessor")
    print("=" * 60)

    # --- Load raw CSVs
    train_df, _           = _load_raw(train_csv, has_label_col=False)
    test_df,  raw_labels  = _load_raw(test_csv,  has_label_col=True)

    # --- Confirm column alignment
    assert list(train_df.columns) == list(test_df.columns), (
        "Train and test sensor columns do not match after dropping prefix and label columns.\n"
        f"Train cols ({len(train_df.columns)}): {list(train_df.columns[:5])} ...\n"
        f"Test  cols ({len(test_df.columns)}):  {list(test_df.columns[:5])} ..."
    )
    print(f"\nSensor columns : {len(train_df.columns)}")

    # --- Cast to numeric (some CSVs export numbers as strings)
    train_df = train_df.apply(pd.to_numeric, errors="coerce")
    test_df  = test_df.apply(pd.to_numeric,  errors="coerce")

    # --- Resolve NaN values
    print("\nResolving NaN values...")
    train_df = _resolve_nan(train_df, "train")
    test_df  = _resolve_nan(test_df,  "test")

    # --- Convert to float32 numpy arrays
    train_data = train_df.to_numpy(dtype=np.float32)
    test_data  = test_df.to_numpy(dtype=np.float32)

    # --- Build label arrays
    train_labels = np.zeros(len(train_data), dtype=int)   # all normal
    test_labels  = _binarise_labels(raw_labels)

    # --- Summary
    print("\n" + "=" * 60)
    print("Output arrays")
    print("=" * 60)
    print(f"  train_data   : {train_data.shape}  dtype={train_data.dtype}")
    print(f"  train_labels : {train_labels.shape}  unique={np.unique(train_labels)}")
    print(f"  test_data    : {test_data.shape}  dtype={test_data.dtype}")
    print(f"  test_labels  : {test_labels.shape}  unique={np.unique(test_labels)}")
    print(f"  Test anomaly rate : {test_labels.mean():.2%}  "
          f"({test_labels.sum()} anomalous timesteps)")

    # Sanity checks
    assert not np.isnan(train_data).any(), "NaN values remain in train_data"
    assert not np.isnan(test_data).any(),  "NaN values remain in test_data"
    assert train_data.shape[1] == test_data.shape[1], "Channel count mismatch"
    print("\nAll sanity checks passed.")

    return train_data, train_labels, test_data, test_labels


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = preprocess_wadi(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
    )

    # --- Plug straight into run_pipeline (no val split needed —
    #     pass test as val too, or hold out a portion of test for calibration)
    # Option A: use test set for both val (threshold calibration) and test eval
    # Option B: split test into val + test (shown below — recommended)

    split_idx    = int(len(test_data) * 0.3)   # first 30% of test → val
    val_data     = test_data[:split_idx]
    val_labels   = test_labels[:split_idx]
    test_data_f  = test_data[split_idx:]
    test_labels_f = test_labels[split_idx:]

    print("\nSplit for run_pipeline:")
    print(f"  val_data   : {val_data.shape}   anomaly_rate={val_labels.mean():.2%}")
    print(f"  test_data  : {test_data_f.shape}  anomaly_rate={test_labels_f.mean():.2%}")

    # --- Now call run_pipeline
    # from temporal_scale_gnn import ScaleGNNConfig, run_pipeline
    #
    # cfg = ScaleGNNConfig(
    #     window_size=12,
    #     stride=1,
    #     hidden_dim=64,
    #     gat_heads=4,
    #     num_gnn_layers=2,
    #     batch_size=256,
    #     num_epochs=30,
    #     learning_rate=1e-3,
    # )
    # model, threshold, metrics = run_pipeline(
    #     train_data,    train_labels,
    #     val_data,      val_labels,
    #     test_data_f,   test_labels_f,
    #     cfg=cfg,
    #     dataset_name="WADI",
    # )
