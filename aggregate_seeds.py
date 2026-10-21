"""
aggregate_seeds.py  —  mean ± std over seeds for CSGL-AD

Reads RESULT_SEEDS/ written by run_seeds.py and produces:
    RESULT_SEEDS/summary.csv          one row per dataset (mean, std, n_seeds)
    RESULT_SEEDS/summary.md           paper-ready table (mean ± std)
    RESULT_SEEDS/per_seed.csv         one row per (dataset, seed) — input for
                                      the significance tests (Welch / Wilcoxon)

SMD: for each seed, the 28 machines are first macro-averaged (as in run.py);
the std reported is then the std of those per-seed averages across seeds.
Peak memory for SMD is the MAX over machines (not the mean / last value),
times are mean per machine.  The across-machine std is also reported
separately (smd_machine_std) for reference.

Std is the sample std (ddof=1), the usual choice for "mean ± std over n runs".

Usage:  python aggregate_seeds.py [--root RESULT_SEEDS]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ["aff_f1", "auroc", "pa_f1", "raw_f1",
           "aff_precision", "aff_recall", "pa_precision", "pa_recall",
           "raw_precision", "raw_recall"]
COSTS = ["train_time_s", "train_time_s_per_epoch", "peak_mem_train_gb",
         "inference_time_s", "inference_ms_per_window", "peak_mem_inference_gb",
         "n_params"]
MAX_OVER_MACHINES = {"peak_mem_train_gb", "peak_mem_inference_gb", "n_params"}
ORDER = ["msl", "smap", "smd", "swat", "wadi", "psm"]
N_SMD_MACHINES = 28


def load(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def per_seed_rows(root: Path):
    rows = []
    for ds_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        ds = ds_dir.name
        if ds == "smd":
            for sd in sorted(ds_dir.glob("seed_*")):
                recs = [load(p) for p in sorted(sd.glob("*.json"))]
                if not recs:
                    continue
                seed = int(sd.name.split("_")[1])
                if len(recs) < N_SMD_MACHINES:
                    print(f"WARNING: smd seed {seed} has only {len(recs)}/"
                          f"{N_SMD_MACHINES} machines — partial average")
                row = {"dataset": ds, "seed": seed, "n_machines": len(recs)}
                for k in METRICS + COSTS:
                    vals = [r["metrics"][k] for r in recs if k in r["metrics"]]
                    if not vals:
                        continue
                    row[k] = float(np.max(vals) if k in MAX_OVER_MACHINES
                                   else np.mean(vals))
                for k in ("aff_f1", "auroc"):
                    row[f"smd_machine_std_{k}"] = float(
                        np.std([r["metrics"][k] for r in recs]))
                rows.append(row)
        else:
            for p in sorted(ds_dir.glob("seed_*.json")):
                r = load(p)
                row = {"dataset": ds, "seed": int(r["seed"])}
                for k in METRICS + COSTS:
                    if k in r["metrics"]:
                        row[k] = float(r["metrics"][k])
                rows.append(row)
    return rows


def summarise(rows):
    out = []
    datasets = sorted({r["dataset"] for r in rows},
                      key=lambda d: ORDER.index(d) if d in ORDER else 99)
    for ds in datasets:
        rs = [r for r in rows if r["dataset"] == ds]
        s = {"dataset": ds, "n_seeds": len(rs),
             "seeds": " ".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))}
        for k in METRICS + COSTS:
            v = np.array([r[k] for r in rs if k in r], dtype=float)
            if v.size == 0:
                continue
            s[f"{k}_mean"] = float(v.mean())
            s[f"{k}_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        out.append(s)
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def pm(s, k, d=3):
    if f"{k}_mean" not in s:
        return "–"
    return f"{s[f'{k}_mean']:.{d}f} ± {s[f'{k}_std']:.{d}f}"


def write_md(path, summ):
    L = ["## Detection (mean ± std over seeds)", "",
         "| Dataset | Seeds | Aff-F1 | AUROC | PA-F1 |",
         "|---|---|---|---|---|"]
    for s in summ:
        L.append(f"| {s['dataset'].upper()} | {s['n_seeds']} | {pm(s,'aff_f1')} | "
                 f"{pm(s,'auroc')} | {pm(s,'pa_f1')} |")
    L += ["", "## Computational cost (mean ± std over seeds; SMD = per machine, "
          "memory = max over machines)", "",
          "| Dataset | Params | Train (s) | s/epoch | Train mem (GB) | "
          "Infer (ms/window) | Infer mem (GB) |",
          "|---|---|---|---|---|---|---|"]
    for s in summ:
        params = f"{int(s['n_params_mean']):,}" if "n_params_mean" in s else "–"
        L.append(f"| {s['dataset'].upper()} | {params} | {pm(s,'train_time_s',1)} | "
                 f"{pm(s,'train_time_s_per_epoch',3)} | {pm(s,'peak_mem_train_gb',3)} | "
                 f"{pm(s,'inference_ms_per_window',3)} | {pm(s,'peak_mem_inference_gb',3)} |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="RESULT_SEEDS")
    try:
        args = ap.parse_args()
    except SystemExit:
        args = ap.parse_args([])
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} not found — run run_seeds.py first.")

    rows = per_seed_rows(root)
    if not rows:
        raise SystemExit("No result files found.")
    summ = summarise(rows)
    write_csv(root / "per_seed.csv", rows)
    write_csv(root / "summary.csv", summ)
    print(write_md(root / "summary.md", summ))
    print(f"\nWrote {root/'per_seed.csv'}, {root/'summary.csv'}, {root/'summary.md'}")


if __name__ == "__main__":
    main()
