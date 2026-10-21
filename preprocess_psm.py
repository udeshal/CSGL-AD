"""
PSM Dataset Preprocessor
========================
Loads the PSM (Pool Server Metrics) dataset from three CSV files:
  - train.csv       : training time series (all normal)
  - test.csv        : test time series (may contain anomalies)
  - test_label.csv  : binary anomaly labels aligned to test.csv by timestamp

Returns numpy arrays ready to pass directly into run_pipeline():
  train_data   : (T_train, C) float32
  train_labels : (T_train,)   int,   all zeros
  test_data    : (T_test, C)  float32
  test_labels  : (T_test,)    int,   binary {0, 1}

NaN handling strategy
---------------------
  1. Forward-fill (propagate last known good value forward).
  2. Backward-fill (catches any leading NaNs at the start of a column).
  3. If an entire column is NaN after both fills, fill with 0.0.

Usage
-----
    from preprocess_psm import load_psm

    train_data, train_labels, test_data, test_labels = load_psm(
        train_path="train.csv",
        test_path="test.csv",
        label_path="test_label.csv",
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def load_psm(
    train_path: str | Path = "train.csv",
    test_path:  str | Path = "test.csv",
    label_path: str | Path = "test_label.csv",
    verbose:    bool  = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the PSM dataset.

    Args:
        train_path : path to train.csv
        test_path  : path to test.csv
        label_path : path to test_label.csv
        verbose    : print shape / NaN / anomaly-rate diagnostics

    Returns:
        train_data, train_labels, test_data, test_labels
    """
    train_path = Path(train_path)
    test_path  = Path(test_path)
    label_path = Path(label_path)

    for p in (train_path, test_path, label_path):
        if not p.exists():
            raise FileNotFoundError(f"PSM file not found: {p.resolve()}")

    # ------------------------------------------------------------------
    # 1. Load CSVs  (row 0 = header, col 0 = timestamp)
    # ------------------------------------------------------------------
    train_df = pd.read_csv(train_path, header=0)
    test_df  = pd.read_csv(test_path,  header=0)
    label_df = pd.read_csv(label_path, header=0)

    if verbose:
        print(f"[PSM] Loaded train : {train_df.shape}  columns={list(train_df.columns[:4])}...")
        print(f"[PSM] Loaded test  : {test_df.shape}")
        print(f"[PSM] Loaded labels: {label_df.shape}")

    # ------------------------------------------------------------------
    # 2. Identify the timestamp column (always the first column)
    # ------------------------------------------------------------------
    ts_col_train = train_df.columns[0]
    ts_col_test  = test_df.columns[0]
    ts_col_label = label_df.columns[0]

    # ------------------------------------------------------------------
    # 3. Align test data with labels via timestamp
    #    Both test.csv and test_label.csv share the same timestamp column.
    #    Inner-join to keep only rows present in both files.
    # ------------------------------------------------------------------
    test_merged = test_df.merge(
        label_df[[ts_col_label, label_df.columns[1]]],   # timestamp + label column
        left_on=ts_col_test,
        right_on=ts_col_label,
        how="inner",
    )

    if verbose:
        print(f"[PSM] After timestamp alignment: test rows = {len(test_merged)} "
              f"(was {len(test_df)})")

    # Identify the label column (last column after merge, or the one from label_df)
    label_col = label_df.columns[1]   # second column in test_label.csv is the binary label

    # Feature columns = everything except timestamp columns and the label column
    drop_cols_test = {ts_col_test, ts_col_label, label_col}
    feat_cols_test = [c for c in test_merged.columns if c not in drop_cols_test]

    # Feature columns for train = everything except the timestamp
    feat_cols_train = [c for c in train_df.columns if c != ts_col_train]

    if verbose:
        print(f"[PSM] Feature columns : {len(feat_cols_train)} "
              f"(train)  /  {len(feat_cols_test)} (test)")
        if set(feat_cols_train) != set(feat_cols_test):
            print("  WARNING: train and test feature columns differ — "
                  "using intersection in original order.")

    # Use intersection of feature columns (preserving train order)
    common_cols = [c for c in feat_cols_train if c in set(feat_cols_test)]

    # ------------------------------------------------------------------
    # 4. Extract raw numpy arrays
    # ------------------------------------------------------------------
    train_raw  = train_df[common_cols].values.astype(np.float32)
    test_raw   = test_merged[common_cols].values.astype(np.float32)
    test_lbls  = test_merged[label_col].values.astype(int)

    if verbose:
        _nan_report("train", train_raw)
        _nan_report("test",  test_raw)

    # ------------------------------------------------------------------
    # 5. NaN imputation  (forward-fill → backward-fill → zero)
    # ------------------------------------------------------------------
    train_raw = _impute(train_raw)
    test_raw  = _impute(test_raw)

    if verbose:
        _nan_report("train (post-impute)", train_raw)
        _nan_report("test  (post-impute)", test_raw)

    # ------------------------------------------------------------------
    # 6. Assign final arrays
    # ------------------------------------------------------------------
    train_data   = train_raw
    train_labels = np.zeros(len(train_data), dtype=int)   # all normal

    test_data    = test_raw
    test_labels  = test_lbls

    # ------------------------------------------------------------------
    # 7. Summary diagnostics
    # ------------------------------------------------------------------
    if verbose:
        C = train_data.shape[1]
        print(f"\n[PSM] Final array shapes")
        print(f"  train_data   : {train_data.shape}   labels: all 0  "
              f"(anomaly_rate={train_labels.mean():.2%})")
        print(f"  test_data    : {test_data.shape}    "
              f"anomaly_rate={test_labels.mean():.2%}")
        print(f"  Channels (C) : {C}")

    return train_data, train_labels, test_data, test_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _impute(arr: np.ndarray) -> np.ndarray:
    """
    NaN imputation on a (T, C) array:
      1. Forward-fill along the time axis (axis=0).
      2. Backward-fill to catch leading NaNs.
      3. Fill any remaining NaN (fully-NaN columns) with 0.0.
    """
    df = pd.DataFrame(arr)
    df = df.ffill(axis=0).bfill(axis=0).fillna(0.0)
    return df.values.astype(np.float32)


def _nan_report(name: str, arr: np.ndarray) -> None:
    """Print NaN diagnostics for an array."""
    total_nan = int(np.isnan(arr).sum())
    cols_with_nan = int(np.isnan(arr).any(axis=0).sum())
    if total_nan > 0:
        print(f"  [NaN] {name}: {total_nan} NaN values across "
              f"{cols_with_nan}/{arr.shape[1]} columns")
    else:
        print(f"  [NaN] {name}: no NaN values found")


# ---------------------------------------------------------------------------
# Quick smoke-test — run as a script to verify your CSV files load correctly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    train_path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    test_path  = sys.argv[2] if len(sys.argv) > 2 else "test.csv"
    label_path = sys.argv[3] if len(sys.argv) > 3 else "test_label.csv"

    train_data, train_labels, test_data, test_labels = load_psm(
        train_path=train_path,
        test_path=test_path,
        label_path=label_path,
        verbose=True,
    )

    print("\n[Smoke test] dtypes and value ranges:")
    for name, arr in [("train_data", train_data), ("test_data", test_data)]:
        print(f"  {name:12s}  dtype={arr.dtype}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}  "
              f"nan={np.isnan(arr).sum()}")

    print("\n[Smoke test] Passed — arrays are ready for run_pipeline().")

    # --- Example: plug straight into run_pipeline ---
    # from temporal_scale_gnn import run_pipeline, ScaleGNNConfig
    # cfg = ScaleGNNConfig(num_epochs=30, batch_size=256)
    # model, threshold, metrics = run_pipeline(
    #     train_data, train_labels,
    #     test_data,  test_labels,   # pass test as val too, or supply a real val set
    #     test_data,  test_labels,
    #     cfg=cfg,
    #     dataset_name="PSM",
    # )
