"""
aggregate_ablation.py  —  ablation table for CSGL-AD with significance tests

Reads
    RESULT_SEEDS/<dataset>/seed_<s>.json                   (full model)
    RESULT_ABLATION/<variant>/<dataset>/seed_<s>.json      (variants)
    (SMD: .../smd/seed_<s>/<machine>.json)
and writes RESULT_ABLATION/ablation.md, ablation.csv.

For every variant × dataset × metric it reports mean ± std over seeds, the change
relative to the full model (Δ = variant − full, so negative = component helps),
and Welch's t-test against the full model on the per-seed values (both sides
now have per-seed results), Holm-corrected across variants within each
dataset/metric.

SMD: if a variant was run on a subset of machines (--smd-machines), the full
model is restricted to the same machines and seeds so the comparison is paired
on identical data.

Usage:  python aggregate_ablation.py [--full RESULT_SEEDS] [--root RESULT_ABLATION]
                                     [--metrics auroc aff_f1]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ORDER = ["full", "no_multiscale",
         "no_temporal_gnn", "no_spatial_gnn", "no_inter_sensor",
         "no_edge_refine", "static_edges", "der_temporal_only", "der_spatial_only",
         "fusion_temporal_only", "fusion_spatial_only", "fusion_no_scale_match",
         "fusion_concat", "fusion_gated",
         "no_mean_head", "no_std_head", "no_corr_head",
         "mean_head_only", "std_head_only", "corr_head_only"]
GROUP = {"no_multiscale": "Multi-scale encoder",
         "no_temporal_gnn": "GNN modules", "no_spatial_gnn": "GNN modules",
         "no_inter_sensor": "GNN modules",
         "no_edge_refine": "Edge refinement", "static_edges": "Edge refinement",
         "der_temporal_only": "Edge refinement", "der_spatial_only": "Edge refinement",
         "fusion_temporal_only": "Fusion", "fusion_spatial_only": "Fusion",
         "fusion_no_scale_match": "Fusion", "fusion_concat": "Fusion",
         "fusion_gated": "Fusion",
         "no_mean_head": "Score heads", "no_std_head": "Score heads",
         "no_corr_head": "Score heads", "mean_head_only": "Score heads",
         "std_head_only": "Score heads", "corr_head_only": "Score heads"}
LABEL = {"full": "CSGL-AD (full)",
         "no_multiscale": "w/o multi-scale encoder (single scale)",
         "no_temporal_gnn": "w/o temporal GNN", "no_spatial_gnn": "w/o spatial GNN",
         "no_inter_sensor": "w/o inter-sensor attention",
         "no_edge_refine": "w/o edge refinement (uniform)",
         "static_edges": "static learned edge weights",
         "der_temporal_only": "refinement on temporal graph only",
         "der_spatial_only": "refinement on spatial graph only",
         "fusion_temporal_only": "temporal branch only",
         "fusion_spatial_only": "spatial branch only",
         "fusion_no_scale_match": "w/o scale-matched fusion",
         "fusion_concat": "concat + MLP fusion", "fusion_gated": "gated fusion",
         "no_mean_head": "w/o mean head", "no_std_head": "w/o std head",
         "no_corr_head": "w/o cosine head", "mean_head_only": "mean head only",
         "std_head_only": "std head only", "corr_head_only": "cosine head only"}


def load_records(root: Path):
    """{dataset: {seed: {machine_or_None: metrics}}}"""
    out = {}
    if not root.exists():
        return out
    for ds_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        ds = ds_dir.name
        if ds == "smd":
            for sd in ds_dir.glob("seed_*"):
                seed = int(sd.name.split("_")[1])
                for f in sd.glob("*.json"):
                    r = json.loads(f.read_text(encoding="utf-8"))
                    out.setdefault(ds, {}).setdefault(seed, {})[f.stem] = r["metrics"]
        else:
            for f in ds_dir.glob("seed_*.json"):
                r = json.loads(f.read_text(encoding="utf-8"))
                out.setdefault(ds, {}).setdefault(int(r["seed"]), {})[None] = r["metrics"]
    return out


def per_seed_values(recs_ds, metric, seeds=None, machines=None):
    """One value per seed (SMD: mean over the given machines)."""
    vals = {}
    for seed, by_m in recs_ds.items():
        if seeds is not None and seed not in seeds:
            continue
        keys = [k for k in by_m if machines is None or k in machines]
        if machines is not None and len(keys) < len(machines):
            continue                       # incomplete seed for this machine set
        v = [by_m[k][metric] for k in keys if metric in by_m[k]]
        if v:
            vals[seed] = float(np.mean(v))
    return vals


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - r) * p[i]))
        out[i] = running
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="RESULT_SEEDS")
    ap.add_argument("--root", default="RESULT_ABLATION")
    ap.add_argument("--metrics", nargs="+", default=["auroc", "aff_f1"])
    try:
        args = ap.parse_args()
    except SystemExit:
        args = ap.parse_args([])

    root = Path(args.root)
    full = load_records(Path(args.full))
    variants = {"full": full}
    if root.exists():
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            variants[d.name] = load_records(d)
    names = sorted(variants, key=lambda v: ORDER.index(v) if v in ORDER else 99)

    rows = []
    for metric in args.metrics:
        for v in names:
            for ds, recs in variants[v].items():
                if ds not in full:
                    continue
                machines = None
                if ds == "smd":
                    machines = set().union(*[set(m) for m in recs.values()])
                x = per_seed_values(recs, metric, machines=machines)
                y = per_seed_values(full[ds], metric, seeds=set(x) if v != "full" else None,
                                    machines=machines)
                if not x or not y:
                    continue
                xv, yv = np.array(list(x.values())), np.array(list(y.values()))
                row = dict(metric=metric, dataset=ds, variant=v,
                           group=GROUP.get(v, ""), label=LABEL.get(v, v),
                           n_seeds=len(xv), mean=xv.mean(),
                           std=xv.std(ddof=1) if len(xv) > 1 else 0.0,
                           full_mean=yv.mean(), delta=xv.mean() - yv.mean(),
                           n_machines=len(machines) if machines else np.nan,
                           p=np.nan)
                if v != "full" and len(xv) > 1 and len(yv) > 1:
                    if np.std(xv) == 0 and np.std(yv) == 0:
                        row["p"] = 1.0 if xv.mean() == yv.mean() else 0.0
                    else:
                        row["p"] = float(stats.ttest_ind(xv, yv, equal_var=False).pvalue)
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No results found.")
    df["p_holm"] = np.nan
    for _, g in df[df.variant != "full"].groupby(["metric", "dataset"]):
        df.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    df["sig"] = df["p_holm"].apply(lambda p: "" if np.isnan(p) else
                                   "***" if p < 1e-3 else "**" if p < 1e-2 else
                                   "*" if p < 0.05 else "")
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "ablation.csv", index=False)

    # ---- markdown: one table per metric, variants × datasets ---------------
    ds_order = [d for d in ["smd", "psm", "msl", "swat", "wadi", "smap"]
                if d in set(df.dataset)]
    L = ["# CSGL-AD ablation (mean ± std over seeds, in %)", "",
         "Δ = variant − full model (negative ⇒ the removed/changed component helps). "
         "Welch's t-test vs the full model, Holm-corrected across variants per "
         "dataset: * p<0.05, ** p<0.01, *** p<0.001.", ""]
    for metric in args.metrics:
        g = df[df.metric == metric]
        L += [f"## {metric.upper()}", "",
              "| Component | Variant | " + " | ".join(d.upper() for d in ds_order) + " | Avg. Δ |",
              "|---|---|" + "---|" * len(ds_order) + "---|"]
        for v in names:
            gv = g[g.variant == v]
            if gv.empty:
                continue
            cells, deltas = [], []
            for d in ds_order:
                r = gv[gv.dataset == d]
                if r.empty:
                    cells.append("–")
                    continue
                r = r.iloc[0]
                cell = f"{100*r['mean']:.2f} ± {100*r['std']:.2f}"
                if v != "full":
                    cell += f" ({100*r.delta:+.2f}{r.sig})"
                    deltas.append(r.delta)
                cells.append(cell)
            avg = f"{100*np.mean(deltas):+.2f}" if deltas else "–"
            L.append(f"| {GROUP.get(v, '')} | {LABEL.get(v, v)} | " + " | ".join(cells)
                     + f" | {avg} |")
        L.append("")
    smd_sub = df[(df.dataset == "smd") & (df.n_machines < 28)]
    if not smd_sub.empty:
        L.append(f"SMD: computed on {int(smd_sub.n_machines.min())}–"
                 f"{int(smd_sub.n_machines.max())} machines; the full model is "
                 "restricted to the same machines.")
    (root / "ablation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {root/'ablation.md'} and {root/'ablation.csv'}")


if __name__ == "__main__":
    main()
