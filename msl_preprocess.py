"""
MSL Dataset Preprocessor
"""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import pandas as pd


def _parse_sequences(raw: str) -> list[tuple[int, int]]:
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Cannot parse anomaly_sequences: {raw!r}") from exc
    result = []
    for pair in parsed:
        if len(pair) != 2:
            raise ValueError(f"Expected [start, end] pair, got: {pair}")
        result.append((int(pair[0]), int(pair[1])))
    return result


def _load_2d(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array in {path.name}, got {arr.shape}")
    return arr


def _impute_nan(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    for c in range(data.shape[1]):
        col = data[:, c]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        fill = float(col[~nan_mask].mean()) if (~nan_mask).any() else 0.0
        col[nan_mask] = fill
        data[:, c] = col
    return data


def load_msl(
    train_dir:   str | Path,
    test_dir:    str | Path,
    anomaly_csv: str | Path,
    spacecraft:  str = "MSL",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dir   = Path(train_dir)
    test_dir    = Path(test_dir)
    anomaly_csv = Path(anomaly_csv)

    for p in (train_dir, test_dir, anomaly_csv):
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

    # ── Parse CSV ──────────────────────────────────────────────────────── #
    df = pd.read_csv(anomaly_csv, header=0)
    df.columns = df.columns.str.strip()
    msl_df = df[df["spacecraft"].str.strip().str.upper() == spacecraft.upper()]
    if msl_df.empty:
        raise ValueError(f"No '{spacecraft}' rows in {anomaly_csv.name}.")

    anomaly_map: dict[str, list[tuple[int, int]]] = {}
    for _, row in msl_df.iterrows():
        cid  = row["chan_id"].strip()
        seqs = _parse_sequences(row["anomaly_sequences"])
        anomaly_map.setdefault(cid, []).extend(seqs)

    print(f"Entities with anomalies in CSV : {sorted(anomaly_map)}")

    # ── Discover files ─────────────────────────────────────────────────── #
    train_files = {f.stem: f for f in sorted(train_dir.glob("*.npy"))}
    test_files  = {f.stem: f for f in sorted(test_dir.glob("*.npy"))}

    if not train_files:
        raise FileNotFoundError(f"No .npy files in: {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No .npy files in: {test_dir}")

    entities = sorted(set(train_files) & set(test_files))
    if not entities:
        raise ValueError("No entity names common to both train and test dirs.")

    print(f"Entities to load (train ∩ test): {entities}")

    # ── Load arrays ────────────────────────────────────────────────────── #
    print(f"\nLoading {len(entities)} entities ...")
    train_blocks: list[np.ndarray] = []
    test_blocks:  list[np.ndarray] = []
    test_lengths: list[int]        = []
    inferred_C:   int | None       = None

    for entity in entities:
        tr = _load_2d(train_files[entity])
        te = _load_2d(test_files[entity])

        for arr, split in [(tr, "train"), (te, "test")]:
            C_file = arr.shape[1]
            if inferred_C is None:
                inferred_C = C_file
                print(f"  Feature channels (C) inferred from first file: {inferred_C}")
            elif C_file != inferred_C:
                raise ValueError(
                    f"Channel mismatch in {split}/{entity}.npy: "
                    f"expected {inferred_C}, got {C_file}."
                )

        train_blocks.append(tr)
        test_blocks.append(te)
        test_lengths.append(te.shape[0])
        print(f"  {entity:>6s}  train {tr.shape}  test {te.shape}")

    # ── Vertical stack ─────────────────────────────────────────────────── #
    train_data = np.vstack(train_blocks).astype(np.float32)
    test_data  = np.vstack(test_blocks).astype(np.float32)

    # ── Labels ─────────────────────────────────────────────────────────── #
    train_labels = np.zeros(train_data.shape[0], dtype=np.int32)
    test_labels  = np.zeros(test_data.shape[0],  dtype=np.int32)

    # Each entity's anomaly indices are LOCAL to its own array.
    # After stacking, entity i starts at cumulative_offset in test_data.
    # We must add this offset to get the correct global row index.
    cumulative_offset = 0
    for entity, entity_len in zip(entities, test_lengths):
        for start, end in anomaly_map.get(entity, []):
            if start >= entity_len:
                print(f"  WARNING [{entity}]: start={start} >= length={entity_len} — skipped.")
                continue
            clamped_end = min(end, entity_len - 1)
            if clamped_end < end:
                print(f"  WARNING [{entity}]: end={end} clamped to {clamped_end}.")

            # Shift local indices to global stacked position
            g_start = cumulative_offset + start
            g_end   = cumulative_offset + clamped_end   # inclusive
            test_labels[g_start : g_end + 1] = 1

        cumulative_offset += entity_len

    # ── NaN imputation ─────────────────────────────────────────────────── #
    nan_tr = int(np.isnan(train_data).sum())
    nan_te = int(np.isnan(test_data).sum())
    if nan_tr > 0:
        print(f"\n  Imputing {nan_tr} NaN(s) in train_data ...")
        train_data = _impute_nan(train_data)
    if nan_te > 0:
        print(f"  Imputing {nan_te} NaN(s) in test_data ...")
        test_data  = _impute_nan(test_data)

    # ── Assertions ─────────────────────────────────────────────────────── #
    assert train_data.ndim == 2
    assert test_data.ndim  == 2
    assert train_data.shape[1] == test_data.shape[1], "C mismatch"
    assert train_data.shape[0] == train_labels.shape[0]
    assert test_data.shape[0]  == test_labels.shape[0]
    assert (train_labels == 0).all(), "train_labels not all zero"
    assert set(np.unique(test_labels)) <= {0, 1}, "test_labels not binary"
    assert not np.isnan(train_data).any(), "NaN remains in train_data"
    assert not np.isnan(test_data).any(),  "NaN remains in test_data"

    # ── Summary ────────────────────────────────────────────────────────── #
    C = train_data.shape[1]
    print(f"\n{'─'*55}")
    print(f"  Entities loaded  : {len(entities)}")
    print(f"  Channels (C)     : {C}  ← from .npy shape, not entity count")
    print(f"  train_data       : {train_data.shape}  dtype={train_data.dtype}")
    print(f"  train_labels     : {train_labels.shape}  all_zero={bool((train_labels==0).all())}")
    print(f"  test_data        : {test_data.shape}  dtype={test_data.dtype}")
    print(f"  test_labels      : {test_labels.shape}  "
          f"anomalies={test_labels.sum()}  ({test_labels.mean()*100:.2f}% anomalous)")
    print(f"{'─'*55}")

    return train_data, train_labels, test_data, test_labels


if __name__ == "__main__":
    TRAIN_DIR   = "MSL/train"
    TEST_DIR    = "MSL/test"
    ANOMALY_CSV = "labeled_anomalies.csv"

    train_data, train_labels, test_data, test_labels = load_msl(
        train_dir=TRAIN_DIR, test_dir=TEST_DIR, anomaly_csv=ANOMALY_CSV
    )

    split        = int(len(train_data) * 0.8)
    val_data     = train_data[split:]
    val_labels   = train_labels[split:]
    train_data   = train_data[:split]
    train_labels = train_labels[:split]

    print("\nAfter validation split:")
    print(f"  train_data   : {train_data.shape}")
    print(f"  train_labels : {train_labels.shape}  all_zero={bool((train_labels==0).all())}")
    print(f"  val_data     : {val_data.shape}")
    print(f"  val_labels   : {val_labels.shape}  all_zero={bool((val_labels==0).all())}")
    print(f"  test_data    : {test_data.shape}")
    print(f"  test_labels  : {test_labels.shape}  anomaly_rate={test_labels.mean():.2%}")
