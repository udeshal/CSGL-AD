"""
SMAP Dataset Preprocessor
==========================

Folder layout expected
-----------------------
    train/                  <- directory of training .npy files
        P-1.npy             <- shape (X, 25)  -- X varies per entity
        S-1.npy
        ...
    test/                   <- directory of test .npy files
        P-1.npy             <- shape (Y, 25)  -- Y varies per entity
        S-1.npy
        ...
    labeled_anomalies.csv   <- anomaly metadata (see below)

SMAP dataset facts
------------------
    54 unique channel/entity identifiers (chan_id).
    Each .npy file has shape (T, 25) -- 25 input features per timestep.
    The number of features (25) is read from the array shape at runtime;
    it is never hardcoded.

    P-2 appears TWICE in labeled_anomalies.csv -- two separate anomaly
    events on the same channel at different time periods. Both sets of
    sequences are merged and applied to P-2's test array.

labeled_anomalies.csv columns used
------------------------------------
    chan_id           -- matches the .npy filename stem (e.g. "P-1" -> "P-1.npy")
    spacecraft        -- "SMAP" or "MSL"; we keep only SMAP rows
    anomaly_sequences -- list of [start, end] index pairs, INCLUSIVE,
                         0-based into that entity's own test array

What this script produces
--------------------------
    train_data   (T_train, 25)  float32   all entities stacked vertically
    train_labels (T_train,)     int32     all zeros  (training = normal)
    test_data    (T_test,  25)  float32   all entities stacked vertically
    test_labels  (T_test,)      int32     0 = normal, 1 = anomalous

Critical offset mapping
------------------------
    The CSV gives anomaly indices LOCAL to each entity's own test array.
    After vertical stacking, entity i starts at row:
        offset_i = sum(test_lengths[0] + ... + test_lengths[i-1])
    Every anomaly index is shifted by this offset before being written
    into test_labels so that labels align correctly with test_data rows.

NaN handling
------------
    Per-column mean imputation is applied after stacking.
    Any column that is entirely NaN is filled with 0.0.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sequences(raw: str) -> list[tuple[int, int]]:
    """
    Parse  "[[2149, 2349], [4536, 4844]]"  ->  [(2149, 2349), (4536, 4844)]
    """
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Cannot parse anomaly_sequences: {raw!r}") from exc
    result = []
    for pair in parsed:
        if len(pair) != 2:
            raise ValueError(f"Expected a [start, end] pair, got: {pair}")
        result.append((int(pair[0]), int(pair[1])))
    return result


def _load_2d(path: Path) -> np.ndarray:
    """
    Load a .npy file and guarantee a 2-D (T, C) float32 array.
    1-D files are reshaped to (T, 1) so the pipeline stays uniform.
    """
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array in {path.name}, got shape {arr.shape}"
        )
    return arr  # (T, C)


def _impute_nan(data: np.ndarray) -> np.ndarray:
    """
    Per-column mean imputation.
    Columns that are entirely NaN are filled with 0.0.
    """
    data = data.copy()
    for c in range(data.shape[1]):
        col      = data[:, c]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        fill = float(col[~nan_mask].mean()) if (~nan_mask).any() else 0.0
        col[nan_mask] = fill
        data[:, c]    = col
    return data


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_smap(
    train_dir:   str | Path,
    test_dir:    str | Path,
    anomaly_csv: str | Path,
    spacecraft:  str = "SMAP",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SMAP data from train/test .npy folders and build label arrays.

    Parameters
    ----------
    train_dir:   Path to the folder containing training .npy files.
    test_dir:    Path to the folder containing test .npy files.
    anomaly_csv: Path to labeled_anomalies.csv.
    spacecraft:  Row filter on the 'spacecraft' column (default "SMAP").

    Returns
    -------
    train_data   : np.ndarray  (T_train, C)  float32  -- C from .npy shape
    train_labels : np.ndarray  (T_train,)    int32    -- all zeros
    test_data    : np.ndarray  (T_test,  C)  float32
    test_labels  : np.ndarray  (T_test,)     int32    -- 0/1
    """
    train_dir   = Path(train_dir)
    test_dir    = Path(test_dir)
    anomaly_csv = Path(anomaly_csv)

    # ------------------------------------------------------------------
    # 1.  Validate paths
    # ------------------------------------------------------------------
    for p in (train_dir, test_dir, anomaly_csv):
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
    if not train_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {train_dir}")
    if not test_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {test_dir}")

    # ------------------------------------------------------------------
    # 2.  Parse the anomaly CSV
    #
    #     Build:  anomaly_map  {chan_id -> [(start, end), ...]}
    #
    #     P-2 appears TWICE in the SMAP section of the CSV -- two
    #     separate anomaly events on the same physical channel.
    #     setdefault + extend merges both rows into one list so both
    #     anomaly ranges are applied when labelling P-2's test array.
    # ------------------------------------------------------------------
    print(f"Reading anomaly metadata : {anomaly_csv}")
    df = pd.read_csv(anomaly_csv, header=0)
    df.columns = df.columns.str.strip()

    smap_df = df[
        df["spacecraft"].str.strip().str.upper() == spacecraft.upper()
    ].copy()

    if smap_df.empty:
        raise ValueError(
            f"No '{spacecraft}' rows found in {anomaly_csv.name}. "
            f"Values present: {df['spacecraft'].unique().tolist()}"
        )

    anomaly_map: dict[str, list[tuple[int, int]]] = {}
    for _, row in smap_df.iterrows():
        cid  = row["chan_id"].strip()
        seqs = _parse_sequences(row["anomaly_sequences"])
        # Use setdefault + extend so duplicate chan_ids (P-2) are merged
        anomaly_map.setdefault(cid, []).extend(seqs)

    unique_ids = sorted(anomaly_map.keys())
    csv_rows   = len(smap_df)
    print(f"  {spacecraft} rows in CSV          : {csv_rows}")
    print(f"  Unique entities with anomalies : {len(unique_ids)}")
    if csv_rows != len(unique_ids):
        dupes = smap_df[smap_df.duplicated("chan_id", keep=False)]["chan_id"].unique()
        print(f"  Duplicate chan_ids (merged)    : {sorted(dupes.tolist())}")

    # ------------------------------------------------------------------
    # 3.  Discover .npy files
    #     Load only entities present in BOTH train and test directories.
    #     Entities in directories but absent from the CSV get zero-only
    #     label arrays (no anomalies claimed for them).
    # ------------------------------------------------------------------
    train_files = {f.stem: f for f in sorted(train_dir.glob("*.npy"))}
    test_files  = {f.stem: f for f in sorted(test_dir.glob("*.npy"))}

    if not train_files:
        raise FileNotFoundError(f"No .npy files found in: {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No .npy files found in: {test_dir}")

    # Entities present in both directories
    common_entities = sorted(set(train_files) & set(test_files))
    if not common_entities:
        raise ValueError(
            "No entity names are common to both train and test directories.\n"
            "Check that .npy filenames match chan_id values (e.g. 'P-1.npy')."
        )

    # Prefer entities also in the CSV; fall back to all common if no match
    csv_common = [e for e in common_entities if e in anomaly_map]
    if csv_common:
        entities = csv_common
        print(f"\n  Entities in both dirs AND CSV  : {len(entities)}")
    else:
        entities = common_entities
        print(f"\n  NOTE: No name match with CSV — loading all "
              f"{len(entities)} common entities (labels will be all-zero).")

    print(f"  Entities to load: {entities}")

    # ------------------------------------------------------------------
    # 4.  Load each entity's 2-D (T, C) arrays
    #     C is read from the first file's shape and cross-checked for
    #     every subsequent file -- it must be identical across all files.
    # ------------------------------------------------------------------
    print(f"\nLoading {len(entities)} entities ...")

    train_blocks : list[np.ndarray] = []
    test_blocks  : list[np.ndarray] = []
    test_lengths : list[int]        = []   # T_test_i per entity (for offset)
    inferred_C   : int | None       = None

    for entity in entities:
        tr = _load_2d(train_files[entity])   # (T_train_i, C)
        te = _load_2d(test_files[entity])    # (T_test_i,  C)

        # Confirm C is the same across every file
        for arr, split in [(tr, "train"), (te, "test")]:
            C_file = arr.shape[1]
            if inferred_C is None:
                inferred_C = C_file
                print(f"  Feature channels (C) inferred from first file : {inferred_C}")
            elif C_file != inferred_C:
                raise ValueError(
                    f"Channel count mismatch in {split}/{entity}.npy — "
                    f"expected {inferred_C} (from first file), got {C_file}."
                )

        train_blocks.append(tr)
        test_blocks.append(te)
        test_lengths.append(te.shape[0])

        # Count anomalous timesteps for this entity (for the log line)
        seqs   = anomaly_map.get(entity, [])
        T_te   = te.shape[0]
        n_anom = sum(
            min(e, T_te - 1) - s + 1
            for s, e in seqs
            if s < T_te
        )
        print(f"  {entity:>5s}  train {tr.shape}  test {te.shape}  "
              f"anomaly sequences={len(seqs)}  "
              f"anomalous timesteps≈{n_anom}")

    # ------------------------------------------------------------------
    # 5.  Vertical stack  ->  (T_total, C)
    # ------------------------------------------------------------------
    train_data = np.vstack(train_blocks).astype(np.float32)
    test_data  = np.vstack(test_blocks).astype(np.float32)

    # ------------------------------------------------------------------
    # 6.  Build label arrays
    # ------------------------------------------------------------------

    # 6a. train_labels: all zeros, one per training timestep
    train_labels = np.zeros(train_data.shape[0], dtype=np.int32)

    # 6b. test_labels: mark anomalous rows using offset-shifted indices
    #
    #     OFFSET LOGIC
    #     ------------
    #     The CSV stores anomaly indices relative to each entity's OWN
    #     test array (0-based). After stacking, entity i starts at global
    #     row `cumulative_offset`. We add this offset to every [start, end]
    #     pair before writing into test_labels.
    #
    #     Example (3 entities, lengths 8505, 7331, 8516):
    #       Entity P-1  offset=    0  local [2149,2349] -> global [2149, 2349]
    #       Entity S-1  offset= 8505  local [5300,5747] -> global [13805,14252]
    #       Entity E-1  offset=15836  local [5000,5030] -> global [20836,20866]

    test_labels       = np.zeros(test_data.shape[0], dtype=np.int32)
    cumulative_offset = 0

    for entity, entity_len in zip(entities, test_lengths):
        seqs = anomaly_map.get(entity, [])

        for start, end in seqs:
            # Validate against actual array length (not CSV num_values)
            if start >= entity_len:
                print(f"  WARNING [{entity}]: anomaly start={start} >= "
                      f"entity length={entity_len} — sequence [{start},{end}] skipped.")
                continue

            clamped_end = min(end, entity_len - 1)
            if clamped_end < end:
                print(f"  WARNING [{entity}]: anomaly end={end} exceeds "
                      f"entity length={entity_len}, clamped to {clamped_end}.")

            # Shift local index -> global stacked index
            global_start = cumulative_offset + start
            global_end   = cumulative_offset + clamped_end   # inclusive

            test_labels[global_start : global_end + 1] = 1

        cumulative_offset += entity_len   # advance for the next entity

    # ------------------------------------------------------------------
    # 7.  NaN imputation
    # ------------------------------------------------------------------
    nan_tr = int(np.isnan(train_data).sum())
    nan_te = int(np.isnan(test_data).sum())

    if nan_tr > 0:
        print(f"\n  Imputing {nan_tr} NaN value(s) in train_data ...")
        train_data = _impute_nan(train_data)
    if nan_te > 0:
        print(f"  Imputing {nan_te} NaN value(s) in test_data ...")
        test_data  = _impute_nan(test_data)

    # ------------------------------------------------------------------
    # 8.  Final sanity checks
    # ------------------------------------------------------------------
    assert train_data.ndim == 2,                             "train_data not 2-D"
    assert test_data.ndim  == 2,                             "test_data not 2-D"
    assert train_data.shape[1] == test_data.shape[1],        "C mismatch train vs test"
    assert train_data.shape[0] == train_labels.shape[0],     "train row/label mismatch"
    assert test_data.shape[0]  == test_labels.shape[0],      "test row/label mismatch"
    assert (train_labels == 0).all(),                        "train_labels not all zero"
    assert set(np.unique(test_labels)) <= {0, 1},            "test_labels not binary"
    assert not np.isnan(train_data).any(),                   "NaN remains in train_data"
    assert not np.isnan(test_data).any(),                    "NaN remains in test_data"

    # ------------------------------------------------------------------
    # 9.  Summary
    # ------------------------------------------------------------------
    C = train_data.shape[1]
    print(f"\n{'─'*60}")
    print(f"  Spacecraft       : {spacecraft}")
    print(f"  Entities loaded  : {len(entities)}")
    print(f"  Channels (C)     : {C}  <- from .npy shape, not entity count")
    print(f"  train_data       : {train_data.shape}   dtype={train_data.dtype}")
    print(f"  train_labels     : {train_labels.shape}  "
          f"all_zero={bool((train_labels == 0).all())}")
    print(f"  test_data        : {test_data.shape}   dtype={test_data.dtype}")
    print(f"  test_labels      : {test_labels.shape}  "
          f"anomalies={test_labels.sum()}  "
          f"({test_labels.mean() * 100:.2f}% anomalous)")
    print(f"{'─'*60}")

    return train_data, train_labels, test_data, test_labels


# ---------------------------------------------------------------------------
# Entry point / example
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -- Set these to your actual paths ------------------------------------
    TRAIN_DIR   = "SMAP/train"
    TEST_DIR    = "SMAP/test"
    ANOMALY_CSV = "labeled_anomalies_smap.csv"
    # ----------------------------------------------------------------------

    train_data, train_labels, test_data, test_labels = load_smap(
        train_dir   = TRAIN_DIR,
        test_dir    = TEST_DIR,
        anomaly_csv = ANOMALY_CSV,
    )

    # Validation split: last 20 % of train (all normal)
    split        = int(len(train_data) * 0.8)
    val_data     = train_data[split:]
    val_labels   = train_labels[split:]
    train_data   = train_data[:split]
    train_labels = train_labels[:split]

    print("\nAfter validation split:")
    print(f"  train_data   : {train_data.shape}")
    print(f"  train_labels : {train_labels.shape}  "
          f"all_zero={bool((train_labels == 0).all())}")
    print(f"  val_data     : {val_data.shape}")
    print(f"  val_labels   : {val_labels.shape}  "
          f"all_zero={bool((val_labels == 0).all())}")
    print(f"  test_data    : {test_data.shape}")
    print(f"  test_labels  : {test_labels.shape}  "
          f"anomaly_rate={test_labels.mean():.2%}")

    # Plug directly into run_pipeline():
    # from temporal_scale_gnn_v3 import run_pipeline, ScaleGNNConfig
    # model, threshold, metrics = run_pipeline(
    #     train_data, train_labels,
    #     val_data,   val_labels,
    #     test_data,  test_labels,
    #     dataset_name="SMAP",
    # )
