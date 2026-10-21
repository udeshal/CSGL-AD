"""
SWaT Dataset Preprocessor
==========================
Loads the SWaT (Secure Water Treatment) dataset from three Excel files:
  - SWaT_Dataset_Normal_v0.xlsx   : first training file  (all Normal)
  - SWaT_Dataset_Normal_v1.xlsx   : second training file (all Normal)
  - SWaT_Dataset_Attack_v0.xlsx   : test file (Normal + Attack rows)

File layout (all three files share the same structure):
  Row 0  : section group headers (P1, P2, … P6) — skipped
  Row 1  : real column headers   (Timestamp, FIT101, … Normal/Attack)
  Col 0  : Timestamp  — dropped (not a sensor feature)
  Col 1…51: 51 numeric sensor/actuator features — kept
  Col 52 : "Normal/Attack" string label — used for test_labels, then dropped

Outputs
-------
  train_data   : (T_train, 51)  float32  — v0 + v1 concatenated
  train_labels : (T_train,)     int      — all zeros (all Normal)
  test_data    : (T_test,  51)  float32
  test_labels  : (T_test,)      int      — 0=Normal, 1=Attack

NaN handling
------------
  1. Forward-fill along the time axis (propagate last good value).
  2. Backward-fill to cover any leading NaNs at the start of a column.
  3. Zero-fill as final fallback for fully-NaN columns.

Usage
-----
    from preprocess_swat import load_swat

    train_data, train_labels, test_data, test_labels = load_swat(
        normal_v0_path="SWaT_Dataset_Normal_v0.xlsx",
        normal_v1_path="SWaT_Dataset_Normal_v1.xlsx",
        attack_v0_path="SWaT_Dataset_Attack_v0.xlsx",
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_swat(
    normal_v0_path: str | Path = "SWaT_Dataset_Normal_v0.xlsx",
    normal_v1_path: str | Path = "SWaT_Dataset_Normal_v1.xlsx",
    attack_v0_path: str | Path = "SWaT_Dataset_Attack_v0.xlsx",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the SWaT dataset.

    Args:
        normal_v0_path : path to SWaT_Dataset_Normal_v0.xlsx
        normal_v1_path : path to SWaT_Dataset_Normal_v1.xlsx
        attack_v0_path : path to SWaT_Dataset_Attack_v0.xlsx
        verbose        : print shape / NaN / anomaly-rate diagnostics

    Returns:
        train_data, train_labels, test_data, test_labels
    """
    normal_v0_path = Path(normal_v0_path)
    normal_v1_path = Path(normal_v1_path)
    attack_v0_path = Path(attack_v0_path)

    for p in (normal_v0_path, normal_v1_path, attack_v0_path):
        if not p.exists():
            raise FileNotFoundError(f"SWaT file not found: {p.resolve()}")

    # ------------------------------------------------------------------
    # 1. Load all three files
    #    header=1  → row index 1 is the true column-header row
    #               (row 0 holds section group labels P1…P6, which we skip)
    # ------------------------------------------------------------------
    if verbose:
        print("[SWaT] Reading Normal_v0 …")
    v0_df = _read_xlsx(normal_v0_path)

    if verbose:
        print("[SWaT] Reading Normal_v1 …")
    v1_df = _read_xlsx(normal_v1_path)

    if verbose:
        print("[SWaT] Reading Attack_v0 …")
    atk_df = _read_xlsx(attack_v0_path)

    if verbose:
        print(f"[SWaT] Raw shapes — v0:{v0_df.shape}  v1:{v1_df.shape}  "
              f"attack:{atk_df.shape}")

    # ------------------------------------------------------------------
    # 2. Identify label column and feature columns
    #    The last column is always "Normal/Attack" (strip whitespace to be safe).
    # ------------------------------------------------------------------
    label_col = atk_df.columns[-1]          # e.g. "Normal/Attack"
    ts_col    = atk_df.columns[0]           # e.g. " Timestamp"

    # Feature columns = everything except timestamp (first) and label (last)
    feat_cols = [c for c in atk_df.columns if c not in (ts_col, label_col)]

    if verbose:
        print(f"[SWaT] Timestamp col : '{ts_col}'")
        print(f"[SWaT] Label col     : '{label_col}'")
        print(f"[SWaT] Feature cols  : {len(feat_cols)}  → {feat_cols[:4]} … {feat_cols[-2:]}")

    # ------------------------------------------------------------------
    # 3. Extract raw feature arrays and label array
    # ------------------------------------------------------------------
    # Ensure feat_cols are present in all three frames (same structure expected)
    for name, df in [("Normal_v0", v0_df), ("Normal_v1", v1_df)]:
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing feature columns: {missing}")

    train_raw_v0 = v0_df[feat_cols].values.astype(np.float32)
    train_raw_v1 = v1_df[feat_cols].values.astype(np.float32)
    test_raw     = atk_df[feat_cols].values.astype(np.float32)

    # Label: "Attack" → 1, everything else ("Normal") → 0
    raw_labels   = atk_df[label_col].astype(str).str.strip()
    test_labels  = (raw_labels == "Attack").astype(int).values

    if verbose:
        _nan_report("Normal_v0 (raw)", train_raw_v0)
        _nan_report("Normal_v1 (raw)", train_raw_v1)
        _nan_report("Attack_v0 (raw)", test_raw)

    # ------------------------------------------------------------------
    # 4. NaN imputation: forward-fill → backward-fill → zero
    # ------------------------------------------------------------------
    train_raw_v0 = _impute(train_raw_v0)
    train_raw_v1 = _impute(train_raw_v1)
    test_raw     = _impute(test_raw)

    if verbose:
        _nan_report("Normal_v0 (post-impute)", train_raw_v0)
        _nan_report("Normal_v1 (post-impute)", train_raw_v1)
        _nan_report("Attack_v0 (post-impute)", test_raw)

    # ------------------------------------------------------------------
    # 5. Concatenate the two training splits
    # ------------------------------------------------------------------
    train_data   = np.concatenate([train_raw_v0, train_raw_v1], axis=0)
    train_labels = np.zeros(len(train_data), dtype=int)

    test_data = test_raw

    # ------------------------------------------------------------------
    # 6. Summary diagnostics
    # ------------------------------------------------------------------
    if verbose:
        C = train_data.shape[1]
        print(f"\n[SWaT] Final array shapes")
        print(f"  train_data   : {train_data.shape}  "
              f"(v0={train_raw_v0.shape[0]} + v1={train_raw_v1.shape[0]} rows)  "
              f"labels: all 0")
        print(f"  test_data    : {test_data.shape}  "
              f"anomaly_rate={test_labels.mean():.2%}  "
              f"({test_labels.sum()} Attack rows)")
        print(f"  Channels (C) : {C}")

    return train_data, train_labels, test_data, test_labels


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_xlsx(path: Path) -> pd.DataFrame:
    """
    Read a SWaT xlsx file with the correct header row.

    Row 0  = section group labels (P1, P2 … P6) — skipped via header=1.
    Row 1  = real column headers  (Timestamp, FIT101, … Normal/Attack).
    header=1 tells pandas to use the second raw row (0-indexed) as names.
    """
    df = pd.read_excel(path, header=1, engine="openpyxl")

    # Strip leading/trailing whitespace from column names and string values
    df.columns = df.columns.str.strip()

    # Strip string values in the label column (last column)
    last_col = df.columns[-1]
    if df[last_col].dtype == object:
        df[last_col] = df[last_col].astype(str).str.strip()

    return df


def _impute(arr: np.ndarray) -> np.ndarray:
    """
    NaN imputation on a (T, C) float32 array:
      1. Forward-fill along time axis (carry last known value forward).
      2. Backward-fill  to handle leading NaNs at the start of a column.
      3. Zero-fill as final fallback for any fully-NaN column.
    """
    df = pd.DataFrame(arr)
    df = df.ffill(axis=0).bfill(axis=0).fillna(0.0)
    return df.values.astype(np.float32)


def _nan_report(name: str, arr: np.ndarray) -> None:
    """Print NaN diagnostics for a (T, C) array."""
    total_nan    = int(np.isnan(arr).sum())
    cols_with_nan = int(np.isnan(arr).any(axis=0).sum())
    if total_nan > 0:
        print(f"  [NaN] {name}: {total_nan} NaN values across "
              f"{cols_with_nan}/{arr.shape[1]} columns")
    else:
        print(f"  [NaN] {name}: no NaN values")


# ---------------------------------------------------------------------------
# Smoke test — run as a script to verify your files load correctly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    normal_v0 = sys.argv[1] if len(sys.argv) > 1 else "SWaT_Dataset_Normal_v0.xlsx"
    normal_v1 = sys.argv[2] if len(sys.argv) > 2 else "SWaT_Dataset_Normal_v1.xlsx"
    attack_v0 = sys.argv[3] if len(sys.argv) > 3 else "SWaT_Dataset_Attack_v0.xlsx"

    train_data, train_labels, test_data, test_labels = load_swat(
        normal_v0_path=normal_v0,
        normal_v1_path=normal_v1,
        attack_v0_path=attack_v0,
        verbose=True,
    )

    print("\n[Smoke test] dtypes and value ranges:")
    for name, arr in [("train_data", train_data), ("test_data", test_data)]:
        print(f"  {name:12s}  dtype={arr.dtype}  shape={arr.shape}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}  "
              f"nan={int(np.isnan(arr).sum())}")

    print(f"\n[Smoke test] Label check:")
    print(f"  train_labels unique : {np.unique(train_labels)}")
    print(f"  test_labels  unique : {np.unique(test_labels)}")
    print(f"  test anomaly rate   : {test_labels.mean():.2%}")

    print("\n[Smoke test] Passed — arrays are ready for run_pipeline().")

    # --- Example: plug straight into run_pipeline ---
    # from temporal_scale_gnn import run_pipeline, ScaleGNNConfig
    # cfg = ScaleGNNConfig(num_epochs=30, batch_size=256)
    # model, threshold, metrics = run_pipeline(
    #     train_data,  train_labels,
    #     test_data,   test_labels,   # used as both val and test in run_pipeline
    #     test_data,   test_labels,
    #     cfg=cfg,
    #     dataset_name="SWaT",
    # )
