"""
run_seeds.py  —  multi-seed runner for CSGL-AD (statistical-significance experiments)

Runs the same model / splits / hyper-parameters as run.py, but repeats each
dataset with several random seeds and saves one small JSON per run so that
(a) results can be aggregated as mean ± std and used in significance tests, and
(b) the script can be re-launched after a Colab disconnect and will skip the
    runs that already finished.

Usage (Colab):
    !python run_seeds.py --dataset-name msl --seeds 0 1 2
    !python run_seeds.py --dataset-name smd --seeds 0 1 2
    !python run_seeds.py --dataset-name all --seeds 0 1 2 3 4
    # ablations
    !python run_seeds.py --list
    !python run_seeds.py --dataset-name msl psm wadi --variant core --seeds 0 1 2

Output layout (inside the Drive project folder, so it survives disconnects):
    RESULT_SEEDS/<dataset>/seed_<s>.json
    RESULT_SEEDS/smd/seed_<s>/<machine>.json
Then run:  !python aggregate_seeds.py
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths — same project folder as run.py
# ---------------------------------------------------------------------------
PROJECT_DIR = "/content/drive/MyDrive/Udesha/ScaledSTGNN_V2/"
os.chdir(PROJECT_DIR)

from Temp_Spatial_Inter_New import ScaleGNNConfig, run_pipeline   # noqa: E402
from DataLoader import load_data                                   # noqa: E402
from smd_preprocess import load_smd                                # noqa: E402

OUT_ROOT = Path("RESULT_SEEDS")
ABL_ROOT = Path("RESULT_ABLATION")

# ---------------------------------------------------------------------------
# Per-dataset settings (taken from run.py and from the logged configs of the
# runs reported in the paper).  Edit here if a dataset used something else.
#   val_split: ("head", f) -> val = first f of test, test = rest
#              ("tail", f) -> test = first f of test, val = rest   (SMAP)
# ---------------------------------------------------------------------------
LONG_WIN  = dict(window_size=60, stride=10, fine_span=10, mid_span=20, coarse_span=60)
SHORT_WIN = dict(window_size=12, stride=2,  fine_span=3,  mid_span=6,  coarse_span=12)

DATASETS = {
    "msl":  dict(win=LONG_WIN,  batch_size=256, lr=2e-3,  val_split=("head", 0.05)),
    "smap": dict(win=LONG_WIN,  batch_size=256, lr=2e-3,  val_split=("tail", 0.60)),
    "wadi": dict(win=LONG_WIN,  batch_size=320, lr=35e-4, val_split=("head", 0.70)),
    "psm":  dict(win=SHORT_WIN, batch_size=256, lr=2e-3,  val_split=("head", 0.70)),
    "swat": dict(win=SHORT_WIN, batch_size=640, lr=5e-3,  val_split=("head", 0.30)),
    # NOTE: check that SMD really used the short window in the paper runs.
    "smd":  dict(win=SHORT_WIN, batch_size=256, lr=2e-3,  val_split=("head", 0.30)),
}

# ---------------------------------------------------------------------------
# Ablation variants: overrides applied on top of the full CSGL-AD config.
# "full" writes to RESULT_SEEDS/ (re-uses the main multi-seed runs);
# every other variant writes to RESULT_ABLATION/<variant>/.
# ---------------------------------------------------------------------------
VARIANTS = {
    "full":               {},
    # Multi-scale encoder
    "no_multiscale":      dict(multiscale_encoder=False),
    # GNN modules
    "no_temporal_gnn":    dict(use_temporal_gnn=False),
    "no_spatial_gnn":     dict(use_spatial_gnn=False),
    "no_inter_sensor":    dict(use_inter_sensor_attn=False),
    # Dynamic edge refinement
    "no_edge_refine":     dict(edge_refinement="none"),
    "static_edges":       dict(edge_refinement="static"),
    "der_temporal_only":  dict(edge_refine_where="temporal"),
    "der_spatial_only":   dict(edge_refine_where="spatial"),
    # Fusion layer
    "fusion_temporal_only": dict(fusion_mode="temporal_only"),
    "fusion_spatial_only":  dict(fusion_mode="spatial_only"),
    "fusion_no_scale_match": dict(fusion_scale_match=False),
    "fusion_concat":      dict(fusion_mode="concat"),
    "fusion_gated":       dict(fusion_mode="gated"),
    # Anomaly-score heads (weights renormalised over the remaining heads)
    "no_mean_head":       dict(score_heads="std,corr"),
    "no_std_head":        dict(score_heads="mean,corr"),
    "no_corr_head":       dict(score_heads="mean,std"),
    "mean_head_only":     dict(score_heads="mean"),
    "std_head_only":      dict(score_heads="std"),
    "corr_head_only":     dict(score_heads="corr"),
}
# Minimal set that answers the reviewer (one or two variants per component)
CORE_VARIANTS = ["no_temporal_gnn", "no_spatial_gnn", "no_edge_refine", "fusion_concat"]
# run the new one on its own with:  --variant no_multiscale

SMD_MACHINES = [f"machine-{g}-{i}"
                for g, count in [(1, 8), (2, 9), (3, 11)]
                for i in range(1, count + 1)]


def make_cfg(ds: str, variant: str = "full") -> ScaleGNNConfig:
    s = DATASETS[ds]
    cfg = ScaleGNNConfig(
        **s["win"],
        hidden_dim=128, gat_heads=4, num_gnn_layers=2,
        reconstruction_hidden=64, fusion_dim=64,
        use_inter_sensor_attn=True,
        inter_sensor_mode="hybrid",
        inter_sensor_top_k=5,
        mean_score_weight=0.3,
        std_score_weight=0.2,
        corr_score_weight=0.5,
        batch_size=s["batch_size"],
        num_epochs=80,
        learning_rate=s["lr"],
        min_precision=0.20,
        fbeta=0.5,
        threshold_percentile=99.0,
    )
    cfg = cfg._replace(**VARIANTS[variant])
    if EPOCHS_OVERRIDE is not None:
        cfg = cfg._replace(num_epochs=EPOCHS_OVERRIDE)
    return cfg


EPOCHS_OVERRIDE = None


def out_root(variant: str) -> Path:
    if variant == "full":
        return OUT_ROOT
    return ABL_ROOT / variant


def split(ds, train_data, train_label, test_data, test_label):
    mode, frac = DATASETS[ds]["val_split"]
    k = int(len(test_data) * frac)
    if mode == "head":
        val_x, val_y = test_data[:k], test_label[:k]
        te_x, te_y = test_data[k:], test_label[k:]
    else:  # "tail"
        te_x, te_y = test_data[:k], test_label[:k]
        val_x, val_y = test_data[k:], test_label[k:]
    return train_data, train_label, val_x, val_y, te_x, te_y


def to_jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, torch.Tensor):
            v = v.item() if v.numel() == 1 else v.tolist()
        out[k] = v
    return out


def save_result(path: Path, ds, seed, cfg, threshold, metrics, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "dataset": ds,
        "seed": int(seed),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold": float(threshold),
        "config": to_jsonable(cfg._asdict()),
        "metrics": to_jsonable(metrics),
    }
    if extra:
        rec.update(extra)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, path)          # atomic: a half-written file is never left behind


def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def summary_line(tag, m):
    return (f"[{tag}] Aff-F1={m['aff_f1']:.4f}  AUROC={m['auroc']:.4f}  "
            f"PA-F1={m['pa_f1']:.4f}  train={m['train_time_s']:.1f}s  "
            f"mem={m['peak_mem_train_gb']:.3f}GB")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_single(ds: str, seeds, force=False, variant="full"):
    out_dir = out_root(variant) / ds
    todo = [s for s in seeds if force or not (out_dir / f"seed_{s}.json").exists()]
    for s in seeds:
        if s not in todo:
            print(f"[{ds}] seed {s}: already done — skipping")
    if not todo:
        return

    train_data, train_label, test_data, test_label = load_data(ds)
    parts = split(ds, train_data, train_label, test_data, test_label)
    cfg = make_cfg(ds, variant)
    print(f"[{ds}|{variant}] train {parts[0].shape}  val {parts[2].shape}  test {parts[4].shape}")

    for s in todo:
        print(f"\n===== {ds.upper()}  {variant}  seed {s} =====")
        tag = "" if variant == "full" else f"_{variant}"
        _model, thr, m = run_pipeline(*parts, cfg=cfg,
                                      dataset_name=f"{ds}{tag}_seed{s}", seed=s)
        save_result(out_dir / f"seed_{s}.json", ds, s, cfg, thr, m,
                    extra={"variant": variant,
                           "val_anomaly_rate": float(np.mean(parts[3])),
                           "test_anomaly_rate": float(np.mean(parts[5]))})
        print(summary_line(f"{ds} {variant} seed {s}", m))
        del _model
        free_gpu()


def run_smd(seeds, force=False, variant="full", machines=None):
    cfg = make_cfg("smd", variant)
    tag = "" if variant == "full" else f"_{variant}"
    for s in seeds:
        out_dir = out_root(variant) / "smd" / f"seed_{s}"
        for machine in (machines or SMD_MACHINES):
            out = out_dir / f"{machine}.json"
            if out.exists() and not force:
                print(f"[smd] seed {s} {machine}: already done — skipping")
                continue
            base = Path("DATA/smd")
            result = load_smd(train_csv=base / "train" / f"{machine}.txt",
                              test_csv=base / "test" / f"{machine}.txt",
                              labels_csv=base / "test_label" / f"{machine}.txt")
            if result is None:
                print(f"[smd] {machine}: files missing — skipping")
                continue
            parts = split("smd", *result)
            print(f"\n===== SMD {machine}  {variant}  seed {s} =====")
            _model, thr, m = run_pipeline(*parts, cfg=cfg,
                                          dataset_name=f"SMD_{machine}{tag}_seed{s}",
                                          seed=s)
            save_result(out, "smd", s, cfg, thr, m,
                        extra={"variant": variant, "machine": machine,
                               "val_has_anomaly": bool(np.sum(parts[3]) > 0),
                               "val_anomaly_rate": float(np.mean(parts[3])),
                               "test_anomaly_rate": float(np.mean(parts[5]))})
            print(summary_line(f"smd seed {s} {machine}", m))
            del _model
            free_gpu()


def main():
    global EPOCHS_OVERRIDE, OUT_ROOT, ABL_ROOT
    p = argparse.ArgumentParser(description="CSGL-AD multi-seed / ablation runner")
    p.add_argument("--dataset-name", nargs="+", default=["msl"],
                   help="one or more of msl smap wadi psm swat smd, or 'all'")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--variant", nargs="+", default=["full"],
                   help="ablation variant(s), 'core' for the core set, 'all' for every "
                        f"variant. Available: {', '.join(VARIANTS)}")
    p.add_argument("--smd-machines", nargs="+", default=None,
                   help="subset of SMD machines (e.g. machine-1-1 machine-2-1)")
    p.add_argument("--force", action="store_true",
                   help="re-run even if a result file already exists")
    p.add_argument("--epochs", type=int, default=None,
                   help="override num_epochs (e.g. 2 for a quick smoke test)")
    p.add_argument("--list", action="store_true", help="print the variants and exit")
    try:
        args = p.parse_args()
    except SystemExit:          # notebook cell without CLI args
        args = p.parse_args([])

    if args.list:
        for k, v in VARIANTS.items():
            print(f"{k:24s} {v}{'   [core]' if k in CORE_VARIANTS else ''}")
        return

    if args.epochs is not None:          # smoke-test mode: separate output folders
        EPOCHS_OVERRIDE = args.epochs
        OUT_ROOT = Path("RESULT_SEEDS_SMOKE")
        ABL_ROOT = Path("RESULT_ABLATION_SMOKE")
        print(f"Smoke test: num_epochs={args.epochs}, writing to {OUT_ROOT}/ and {ABL_ROOT}/")

    variants = []
    for v in args.variant:
        variants += CORE_VARIANTS if v == "core" else list(VARIANTS) if v == "all" else [v]
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"Unknown variant '{v}'. Use --list to see the options.")

    names = [d.lower() for d in args.dataset_name]
    datasets = list(DATASETS) if "all" in names else names
    for ds in datasets:
        if ds not in DATASETS:
            raise ValueError(f"Unknown dataset '{ds}'. Choose from {list(DATASETS)} or 'all'.")

    for v in variants:                   # variant-major: one component finished at a time
        for ds in datasets:
            if ds == "smd":
                run_smd(args.seeds, args.force, v, args.smd_machines)
            else:
                run_single(ds, args.seeds, args.force, v)
    print("\nDone. Aggregate with:  python aggregate_seeds.py   (main runs)"
          "\n                       python aggregate_ablation.py (ablation table)")


if __name__ == "__main__":
    main()
