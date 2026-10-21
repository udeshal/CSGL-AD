"""
SMD Dataset Preprocessor
=========================
Loads the SMD (Server Machine Dataset) from three folders:
  train/        — one .txt file per machine, comma-separated, no header
  test/         — matching .txt files, same format
  test_label/   — matching .txt files, one integer (0 or 1) per line

For each machine file the three files must share the same filename stem,
e.g.:
  train/machine-3-9.txt
  test/machine-3-9.txt
  test_label/machine-3-9.txt

All matching machine files are loaded and concatenated (in sorted filename
order) into four numpy arrays:

  train_data   : (T_train, C)  float32  — all machines, train split
  train_labels : (T_train,)    int      — all zeros (all Normal)
  test_data    : (T_test,  C)  float32  — all machines, test split
  test_labels  : (T_test,)     int      — 0=Normal, 1=Anomaly

NaN handling
------------
  1. Forward-fill along the time axis (propagate last known good value).
  2. Backward-fill to cover any leading NaNs at the start of a column.
  3. Zero-fill as final fallback for any fully-NaN column.

File format (confirmed from sample files)
-----------------------------------------
  - No header row.
  - Values are comma-separated floats.
  - One timestep per line.
  - Labels file has one integer (0 or 1) per line, matching test rows 1-to-1.
  - 38 feature columns per row (may vary across datasets — inferred at load time).

Usage
-----
    from preprocess_smd import load_smd

    # Load all machines in the three folders
    train_data, train_labels, test_data, test_labels = load_smd(
        train_dir="train",
        test_dir="test",
        label_dir="test_label",
    )

    # Load a specific subset of machines only
    train_data, train_labels, test_data, test_labels = load_smd(
        train_dir="train",
        test_dir="test",
        label_dir="test_label",
        machine_names=["machine-1-1", "machine-2-3"],
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_smd(
    train_dir:     str | Path = "train",
    test_dir:      str | Path = "test",
    label_dir:     str | Path = "test_label",
    machine_names: list[str] | None = None,
    verbose:       bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the SMD dataset from three folders.

    Args:
        train_dir     : folder containing train .txt files
        test_dir      : folder containing test  .txt files
        label_dir     : folder containing test_label .txt files
        machine_names : optional list of machine name stems to load
                        (e.g. ["machine-1-1", "machine-2-3"]).
                        If None, all files found in train_dir are loaded.
        verbose       : print per-machine and summary diagnostics

    Returns:
        train_data, train_labels, test_data, test_labels
    """
    train_dir = Path(train_dir)
    test_dir  = Path(test_dir)
    label_dir = Path(label_dir)

    for d in (train_dir, test_dir, label_dir):
        if not d.exists():
            raise FileNotFoundError(f"SMD directory not found: {d.resolve()}")

    # ------------------------------------------------------------------
    # 1. Discover machine files
    # ------------------------------------------------------------------
    # Collect all file stems from train_dir (sorted for reproducibility)
    all_stems = sorted(p.stem for p in train_dir.iterdir()
                       if p.suffix in (".txt", ".csv") and not p.name.startswith("."))

    if not all_stems:
        raise FileNotFoundError(
            f"No .txt / .csv files found in train_dir: {train_dir.resolve()}"
        )

    if machine_names is not None:
        # Validate requested names
        missing = [m for m in machine_names if m not in all_stems]
        if missing:
            raise ValueError(
                f"Requested machines not found in train_dir: {missing}\n"
                f"Available: {all_stems}"
            )
        stems_to_load = sorted(machine_names)
    else:
        stems_to_load = all_stems

    if verbose:
        print(f"[SMD] Found {len(all_stems)} machine file(s) in train_dir.")
        print(f"[SMD] Loading {len(stems_to_load)} machine(s): {stems_to_load}")

    # ------------------------------------------------------------------
    # 2. Load each machine and accumulate arrays
    # ------------------------------------------------------------------
    train_parts: list[np.ndarray] = []
    test_parts:  list[np.ndarray] = []
    label_parts: list[np.ndarray] = []

    for stem in stems_to_load:
        train_file = _find_file(train_dir, stem)
        test_file  = _find_file(test_dir,  stem)
        label_file = _find_file(label_dir, stem)

        if verbose:
            print(f"\n  [{stem}]")

        train_arr = _load_data_file(train_file, verbose=verbose, tag="train")
        test_arr  = _load_data_file(test_file,  verbose=verbose, tag="test ")
        label_arr = _load_label_file(label_file, verbose=verbose)

        # Sanity: test rows must match label rows
        if len(test_arr) != len(label_arr):
            raise ValueError(
                f"[{stem}] test rows ({len(test_arr)}) != label rows ({len(label_arr)})"
            )

        train_parts.append(train_arr)
        test_parts.append(test_arr)
        label_parts.append(label_arr)

    # ------------------------------------------------------------------
    # 3. Concatenate across machines
    # ------------------------------------------------------------------
    train_data  = np.concatenate(train_parts, axis=0)
    test_data   = np.concatenate(test_parts,  axis=0)
    test_labels = np.concatenate(label_parts, axis=0)

    # ------------------------------------------------------------------
    # 4. NaN imputation (after concatenation so fill can cross boundaries)
    # ------------------------------------------------------------------
    train_data = _impute(train_data)
    test_data  = _impute(test_data)

    # ------------------------------------------------------------------
    # 5. Build train_labels (all zeros — SMD train split is all normal)
    # ------------------------------------------------------------------
    train_labels = np.zeros(len(train_data), dtype=int)

    # ------------------------------------------------------------------
    # 6. Summary diagnostics
    # ------------------------------------------------------------------
    if verbose:
        C = train_data.shape[1]
        print(f"\n[SMD] Final array shapes")
        print(f"  train_data   : {train_data.shape}   labels: all 0")
        print(f"  test_data    : {test_data.shape}   "
              f"anomaly_rate={test_labels.mean():.2%}  "
              f"({test_labels.sum()} anomalous rows)")
        print(f"  Channels (C) : {C}")
        _nan_report("train_data (final)", train_data)
        _nan_report("test_data  (final)", test_data)

    return train_data, train_labels, test_data, test_labels


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_file(directory: Path, stem: str) -> Path:
    """
    Locate a file in `directory` whose stem matches `stem`.
    Accepts both .txt and .csv extensions.
    Raises FileNotFoundError if not found.
    """
    for ext in (".txt", ".csv"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find '{stem}.txt' or '{stem}.csv' in {directory.resolve()}"
    )


def _load_data_file(path: Path, verbose: bool, tag: str) -> np.ndarray:
    """
    Load a comma-separated SMD data file (no header) into a float32 array.
    Returns shape (T, C).
    """
    df  = pd.read_csv(path, header=None, sep=",")
    arr = df.values.astype(np.float32)
    if verbose:
        nan_count = int(np.isnan(arr).sum())
        print(f"    {tag}: shape={arr.shape}  NaN={nan_count}")
    return arr


def _load_label_file(path: Path, verbose: bool) -> np.ndarray:
    """
    Load an SMD label file — one integer (0 or 1) per line.
    Returns shape (T,) int array.
    """
    labels = pd.read_csv(path, header=None, sep=",").iloc[:, 0].values.astype(int)
    if verbose:
        print(f"    label: shape={labels.shape}  "
              f"anomaly_rate={labels.mean():.2%}  unique={np.unique(labels).tolist()}")
    return labels


def _impute(arr: np.ndarray) -> np.ndarray:
    """
    NaN imputation on a (T, C) float32 array:
      1. Forward-fill along time axis (carry last known value forward).
      2. Backward-fill to handle leading NaNs at the start of a column.
      3. Zero-fill as final fallback for any fully-NaN column.
    """
    df = pd.DataFrame(arr)
    df = df.ffill(axis=0).bfill(axis=0).fillna(0.0)
    return df.values.astype(np.float32)


def _nan_report(name: str, arr: np.ndarray) -> None:
    """Print NaN count for an array."""
    total = int(np.isnan(arr).sum())
    if total > 0:
        cols = int(np.isnan(arr).any(axis=0).sum())
        print(f"  [NaN] {name}: {total} NaN across {cols}/{arr.shape[1]} columns")
    else:
        print(f"  [NaN] {name}: no NaN values")


# ---------------------------------------------------------------------------
# Smoke test — run as a script to verify your folders load correctly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    train_dir = sys.argv[1] if len(sys.argv) > 1 else "train"
    test_dir  = sys.argv[2] if len(sys.argv) > 2 else "test"
    label_dir = sys.argv[3] if len(sys.argv) > 3 else "test_label"

    train_data, train_labels, test_data, test_labels = load_smd(
        train_dir=train_dir,
        test_dir=test_dir,
        label_dir=label_dir,
        verbose=True,
    )

    print("\n[Smoke test] dtypes and value ranges:")
    for name, arr in [("train_data", train_data), ("test_data", test_data)]:
        print(f"  {name:12s}  dtype={arr.dtype}  shape={arr.shape}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  "
              f"nan={int(np.isnan(arr).sum())}")

    print(f"\n[Smoke test] Label check:")
    print(f"  train_labels unique : {np.unique(train_labels).tolist()}")
    print(f"  test_labels  unique : {np.unique(test_labels).tolist()}")
    print(f"  test anomaly rate   : {test_labels.mean():.2%}")

    print("\n[Smoke test] Passed — arrays are ready for run_pipeline().")

    # --- Example: plug straight into run_pipeline ---
    # from temporal_scale_gnn import run_pipeline, ScaleGNNConfig
    # cfg = ScaleGNNConfig(num_epochs=30, batch_size=256)
    # model, threshold, metrics = run_pipeline(
    #     train_data,  train_labels,
    #     test_data,   test_labels,
    #     test_data,   test_labels,
    #     cfg=cfg,
    #     dataset_name="SMD",
    # )
