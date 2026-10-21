"""
Spatio-Temporal Scale GNN for Anomaly Detection  — v3
=======================================================
Diagnosis of v2 failure
------------------------
v2 introduced a full-sequence reconstruction decoder (span×C outputs per node).
For WADI with C=127 channels, this meant the coarse decoder had to output
12×127=1524 values from a 64-dim embedding — a >23× expansion ratio.  The
decoder had no chance to learn: loss plateaued at 0.136 from epoch 1 (random
baseline) and gradients were uniformly uninformative.  Consequently:

  - AUROC dropped from 0.905 → 0.752 (worse than v1)
  - z-normalised val scores had mean=-0.845 (score ordering inverted)
  - Threshold went negative → flagged everything → F1=0.133 again

v3 strategy: KEEP THE ARCHITECTURE THAT GAVE AUROC=0.905, fix only the
two things that were actually broken:

  1. Threshold selection  (the real cause of F1 collapse in v1)
  2. Score separation     (widen the gap between normal and anomaly scores)

What v3 changes vs v1
----------------------
CHANGE-A  Multi-head anomaly score  (replaces single window-mean target)
          Instead of one MSE head, we compute THREE complementary scores:
            • mean_error:  MSE between reconstructed mean and actual mean
                           (same as v1 — preserves AUROC=0.905 baseline)
            • std_error:   MSE between reconstructed std and actual std
                           (anomalies have unusual variance — lightweight head)
            • corr_error:  1 - cosine_similarity between reconstruction and target
                           (catches directional/pattern deviations)
          All three targets are O(C) — no expansion ratio problem.
          Combined score = weighted sum (weights in cfg).

CHANGE-B  REMOVED — reg_loss_weight was removed after analysis showed it
          actively hurt score separation by suppressing the coarse-node
          max score (the most discriminative signal) on normal windows,
          dragging anomaly scores down alongside normal ones.

CHANGE-C  Robust threshold calibration with precision guard (5 strategies)
          Five strategies evaluated on the validation set:
            A. F1-optimal on PR curve (same as v1 — tends to recall=1)
            B. Fβ-optimal (β=0.5, precision-weighted)
            C. Precision-floor: best F1 subject to P ≥ min_precision
            D. p99 of normal training scores (anchored to normal distribution)
            E. Midpoint between normal-p99 and anomaly-median
          Winner = highest val F1 among strategies with P ≥ min_precision.
          Fallback = strategy D if none meet the floor.
          Also prints score gap diagnostics to confirm model discriminates.

CHANGE-D  No z-score normalisation of scores
          v2's z-normalisation inverted the score ordering because the
          training score mean was higher than anomaly scores (model learned
          to reconstruct anomalies as well as normals — a sign the task was
          too easy or the scores were noise).  We remove it.  Thresholds are
          now raw reconstruction errors, same as v1 but selected more carefully.

Architecture (unchanged from v1 — proven to give AUROC=0.905)
--------------------------------------------------------------
  MultiScaleEncoder → (Temporal GNN  ‖  Spatial GNN) → FusionLayer → AnomalyScorer
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv
from typing import NamedTuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Seed every random number generator the pipeline uses, so a run can be
    repeated with the same seed.

    Covers Python's `random`, NumPy, PyTorch (CPU and all GPUs) and the
    DataLoader shuffle order (which draws from PyTorch's global generator).
    cuDNN is put in deterministic mode. Note: torch_geometric's GATv2 uses
    scatter operations whose CUDA kernels use atomic adds, so GPU runs with
    the same seed can still differ in the last decimal places. Results are
    reproducible to within that small numerical noise, not bit-for-bit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ScaleGNNConfig(NamedTuple):
    """All hyper-parameters in one place.

    num_features is set automatically from training data shape in run_pipeline().
    """
    window_size: int = 12
    stride: int = 1
    fine_span: int = 3
    mid_span: int = 6
    coarse_span: int = 12
    num_features: int = 1           # overwritten at runtime
    hidden_dim: int = 64
    gat_heads: int = 4
    gat_dropout: float = 0.1
    edge_temperature: float = 1.0
    num_gnn_layers: int = 2
    reconstruction_hidden: int = 64
    fusion_dim: int = 64
    # Inter-sensor graph attention (spatial stream, within-scale channel<->channel)
    use_inter_sensor_attn: bool = True
    inter_sensor_mode: str = "hybrid"    # "learned" | "correlation" | "hybrid"
    inter_sensor_top_k: int = 5          # sensors each sensor attends to
    inter_sensor_layers: int = 1         # stacked GATv2 layers
    inter_sensor_embed_dim: int = 16     # size of learned sensor embeddings
    inter_sensor_alpha: float = 3.0      # tanh saturation (MTGNN-style)
    inter_sensor_gate_init: float = 0.0  # 0 => starts exactly at the no-attention baseline
    # CHANGE-A: multi-head score weights (must sum to 1)
    mean_score_weight: float = 0.5   # MSE on window mean  (v1 baseline signal)
    std_score_weight:  float = 0.3   # MSE on window std   (variance anomalies)
    corr_score_weight: float = 0.2   # 1 - cosine sim      (pattern anomalies)
    # Training
    batch_size: int = 256
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # CHANGE-C: threshold calibration
    min_precision: float = 0.20      # precision floor enforced during calibration
    fbeta: float = 0.5               # β for Fβ-optimal strategy
    threshold_percentile: float = 99.0
    # Anomaly decision (overwritten by calibrate_threshold)
    anomaly_threshold: float = 0.5
    # ---- Ablation switches (defaults = full CSGL-AD) ----------------------
    multiscale_encoder: bool = True      # False: single-scale — mid/coarse levels use the fine span
    use_temporal_gnn: bool = True        # False: temporal node features bypass the temporal GNN
    use_spatial_gnn: bool = True         # False: spatial node features bypass the cross-scale spatial GNN
    edge_refinement: str = "dynamic"     # "dynamic" | "static" (learned, input-independent) | "none" (all 1)
    edge_refine_where: str = "both"      # "both" | "temporal" | "spatial"  (other graph gets weights = 1)
    fusion_mode: str = "add"             # "add" | "concat" | "gated" | "temporal_only" | "spatial_only"
    fusion_scale_match: bool = True      # False: every temporal node gets the scale-averaged spatial context
    score_heads: str = "mean,std,corr"   # active anomaly-score heads; weights renormalised over active ones


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    def __init__(self, data, labels, window_size, stride=1):
        super().__init__()
        T = data.shape[0]
        self.windows, self.window_labels = [], []
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            self.windows.append(data[start:end])
            self.window_labels.append(
                int(labels[start:end].max()) if labels is not None else -1
            )
        self.has_labels = labels is not None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.window_labels[idx], dtype=torch.long)
        return x, y


def prepare_splits(train_data, train_labels, val_data, val_labels,
                   test_data, test_labels, cfg, clip_sigma=6.0):
    scaler = StandardScaler()
    train_norm = np.clip(scaler.fit_transform(train_data), -clip_sigma, clip_sigma)
    val_norm   = np.clip(scaler.transform(val_data),       -clip_sigma, clip_sigma)
    test_norm  = np.clip(scaler.transform(test_data),      -clip_sigma, clip_sigma)

    for name, arr in [("train", train_norm), ("val", val_norm), ("test", test_norm)]:
        print(f"  {name:5s} normalised — min={arr.min():.3f}  max={arr.max():.3f}  "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}")

    train_ds = SlidingWindowDataset(train_norm, train_labels, cfg.window_size, cfg.stride)
    val_ds   = SlidingWindowDataset(val_norm,   val_labels,   cfg.window_size, cfg.stride)
    test_ds  = SlidingWindowDataset(test_norm,  test_labels,  cfg.window_size, cfg.stride)

    class_counts = np.bincount(np.array(train_ds.window_labels))

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True)

    return (make_loader(train_ds, True), make_loader(val_ds, False),
            make_loader(test_ds, False), class_counts)


# ===========================================================================
# GRAPH CONSTRUCTION
# ===========================================================================

def build_scale_metadata(cfg):
    node_idx = 0
    scales = {}
    for name, span in [("fine", cfg.fine_span), ("mid", cfg.mid_span),
                        ("coarse", cfg.coarse_span)]:
        count = cfg.window_size // span
        entries = []
        for k in range(count):
            start = k * span
            entries.append((node_idx, start, start + span))
            node_idx += 1
        scales[name] = entries
    return scales


def build_temporal_edge_index(scale_meta):
    edges = []

    def overlaps(s_a, e_a, s_b, e_b):
        return s_a < e_b and s_b < e_a

    for fi, f_s, f_e in scale_meta["fine"]:
        for mi, m_s, m_e in scale_meta["mid"]:
            if overlaps(f_s, f_e, m_s, m_e):
                edges += [(fi, mi), (mi, fi)]
    for mi, m_s, m_e in scale_meta["mid"]:
        for ci, c_s, c_e in scale_meta["coarse"]:
            if overlaps(m_s, m_e, c_s, c_e):
                edges += [(mi, ci), (ci, mi)]
    for nodes in scale_meta.values():
        for idx, _, _ in nodes:
            edges.append((idx, idx))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_temporal_scale_labels(scale_meta):
    labels = []
    for level, name in enumerate(["fine", "mid", "coarse"]):
        labels.extend([level] * len(scale_meta[name]))
    return torch.tensor(labels, dtype=torch.long)


def build_spatial_graph(num_channels):
    C = num_channels
    edges = []
    for k in range(C):
        fn, mn, cn = k, C + k, 2 * C + k
        edges += [(fn, mn), (mn, fn), (mn, cn), (cn, mn),
                  (fn, fn), (mn, mn), (cn, cn)]
    edge_index   = torch.tensor(edges, dtype=torch.long).t().contiguous()
    scale_labels = torch.tensor([0]*C + [1]*C + [2]*C, dtype=torch.long)
    return edge_index, scale_labels


# ===========================================================================
# ENCODER
# ===========================================================================

class TemporalNodeEncoder(nn.Module):
    def __init__(self, span, num_features, hidden_dim):
        super().__init__()
        self.conv = nn.Conv1d(num_features, hidden_dim,
                              kernel_size=min(span, 3), padding="same")
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, window):
        return self.norm(F.gelu(self.conv(window.transpose(1, 2))).mean(dim=-1))


class MultiScaleEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({
            "fine":   TemporalNodeEncoder(cfg.fine_span,   cfg.num_features, cfg.hidden_dim),
            "mid":    TemporalNodeEncoder(cfg.mid_span,    cfg.num_features, cfg.hidden_dim),
            "coarse": TemporalNodeEncoder(cfg.coarse_span, cfg.num_features, cfg.hidden_dim),
        })
        self.channel_proj = nn.ModuleDict({
            "fine":   nn.Linear(cfg.fine_span,   cfg.hidden_dim),
            "mid":    nn.Linear(cfg.mid_span,    cfg.hidden_dim),
            "coarse": nn.Linear(cfg.coarse_span, cfg.hidden_dim),
        })
        self.channel_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x, scale_meta):
        batch_size = x.shape[0]
        temporal_embeddings = []
        spatial_scale_feats = {}

        for scale_name, encoder in self.encoders.items():
            windows_at_scale = []
            for _, start, end in scale_meta[scale_name]:
                emb = encoder(x[:, start:end, :])
                temporal_embeddings.append(emb)
                windows_at_scale.append(x[:, start:end, :])

            stacked  = torch.stack(windows_at_scale, dim=0).mean(dim=0)
            ch_input = stacked.permute(0, 2, 1)
            ch_emb   = self.channel_norm(F.gelu(self.channel_proj[scale_name](ch_input)))
            spatial_scale_feats[scale_name] = ch_emb

        stacked_t = torch.stack(temporal_embeddings, dim=0)
        temporal_node_feats = stacked_t.permute(1, 0, 2).reshape(
            batch_size * len(temporal_embeddings), -1
        )
        return temporal_node_feats, spatial_scale_feats


class SpatialNodeAssembler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_channels = cfg.num_features

    def forward(self, spatial_scale_feats):
        batch_size = next(iter(spatial_scale_feats.values())).shape[0]
        C = self.num_channels
        parts = [spatial_scale_feats["fine"],
                 spatial_scale_feats["mid"],
                 spatial_scale_feats["coarse"]]
        return torch.cat(parts, dim=1).reshape(batch_size * 3 * C, -1)


# ===========================================================================
# GNN MODULES
# ===========================================================================

class DynamicEdgeRefinement(nn.Module):
    def __init__(self, hidden_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feats, edge_index):
        src = node_feats[edge_index[0]]
        dst = node_feats[edge_index[1]]
        logits = self.edge_mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits / self.temperature)


class StaticEdgeWeights(nn.Module):
    """Ablation: one learnable weight per edge of the (unbatched) graph, shared
    by every window, i.e. learned but NOT input-dependent."""
    def __init__(self, num_edges):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_edges))

    def forward(self, batch_size):
        return torch.sigmoid(self.logits).repeat(batch_size)


class ScaleAwareGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale_embed = nn.Embedding(3, cfg.hidden_dim)
        self.gat_layers  = nn.ModuleList([
            GATv2Conv(cfg.hidden_dim, cfg.hidden_dim // cfg.gat_heads,
                      heads=cfg.gat_heads, concat=True, dropout=cfg.gat_dropout,
                      edge_dim=1, add_self_loops=False)
            for _ in range(cfg.num_gnn_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.num_gnn_layers)
        ])

    def forward(self, node_feats, edge_index, edge_weights, scale_labels):
        h = node_feats + self.scale_embed(scale_labels)
        edge_attr = edge_weights.unsqueeze(-1)
        for gat, norm in zip(self.gat_layers, self.norms):
            h = norm(gat(h, edge_index, edge_attr=edge_attr) + h)
        return h


def build_sensor_prior(train_data):
    """
    Dense |Pearson correlation| matrix between sensors, [C, C], zero diagonal.

    Fit on training data only (no leakage). Pearson correlation is invariant to
    per-channel scaling/shifting, so it does not matter that this runs on the
    raw array before StandardScaler. Constant channels (zero variance) give NaN
    correlations; these are mapped to 0 ("correlated with nobody").
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(train_data.T)
    corr = np.abs(np.nan_to_num(corr, nan=0.0))
    np.fill_diagonal(corr, 0.0)
    return torch.tensor(corr, dtype=torch.float32)


class InterSensorAttention(nn.Module):
    """
    Inter-sensor graph attention.

    Nodes are the (scale, channel) nodes of the spatial graph. Edges connect
    DIFFERENT sensors WITHIN the same scale block, so it complements the
    existing spatial GNN (which only links the same sensor across scales).
    A GATv2 layer then learns, per node and per head, which of its neighbouring
    sensors to attend to.

    Where the sensor graph comes from (cfg.inter_sensor_mode):
      "learned"     : MTGNN-style directed adjacency from two learnable sensor
                      embeddings, A = relu(tanh(a (M1 M2^T - M2 M1^T))).
                      Trained end-to-end through the edge weights.
      "correlation" : fixed |Pearson corr| prior computed from training data.
      "hybrid"      : learned adjacency + correlation prior, summed before
                      top-k, so the model starts from a data-driven structure
                      and can refine it.

    Each sensor keeps its top-k strongest sources (fixed in-degree k, no
    isolated nodes, O(C*k) edges instead of O(C^2)). Edge strength is passed to
    GATv2 as an edge feature, so the model sees HOW strong each dependency is,
    not just whether it exists.

    Over-smoothing guard: the update is  h <- LayerNorm(h + g * GAT(h))  with a
    learnable scalar gate g per layer (init cfg.inter_sensor_gate_init, default
    0). The module therefore starts as an exact identity and only admits
    cross-sensor information to the extent that it lowers the training loss.
    This matters for reconstruction-based detection: letting a normal neighbour
    "explain away" an anomalous sensor would shrink the very error we score on.

    Edge direction: row i of the adjacency is the RECEIVING sensor; its top-k
    columns j are the sources it attends to, i.e. edges j -> i.
    """

    MODES = ("learned", "correlation", "hybrid")

    def __init__(self, cfg, sensor_prior=None):
        super().__init__()
        if cfg.inter_sensor_mode not in self.MODES:
            raise ValueError(f"inter_sensor_mode must be one of {self.MODES}, "
                             f"got {cfg.inter_sensor_mode!r}")
        C = cfg.num_features
        if C < 2:
            raise ValueError("InterSensorAttention needs at least 2 sensors.")

        self.C          = C
        self.mode       = cfg.inter_sensor_mode
        self.use_learned = self.mode in ("learned", "hybrid")
        self.use_prior   = self.mode in ("correlation", "hybrid")
        self.k          = max(1, min(cfg.inter_sensor_top_k, C - 1))
        self.alpha      = cfg.inter_sensor_alpha

        if self.use_learned:
            self.emb_a = nn.Embedding(C, cfg.inter_sensor_embed_dim)
            self.emb_b = nn.Embedding(C, cfg.inter_sensor_embed_dim)
        if self.use_prior:
            if sensor_prior is None or tuple(sensor_prior.shape) != (C, C):
                raise ValueError(
                    f"mode={self.mode!r} needs sensor_prior of shape ({C}, {C}); "
                    f"build it with build_sensor_prior(train_data).")
            self.register_buffer("prior", sensor_prior.float())

        self.register_buffer("diag_mask", torch.eye(C, dtype=torch.bool))

        self.gat_layers = nn.ModuleList([
            GATv2Conv(cfg.hidden_dim, cfg.hidden_dim // cfg.gat_heads,
                      heads=cfg.gat_heads, concat=True, dropout=cfg.gat_dropout,
                      edge_dim=1, add_self_loops=False)
            for _ in range(cfg.inter_sensor_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.inter_sensor_layers)
        ])
        self.gates = nn.ParameterList([
            nn.Parameter(torch.full((1,), float(cfg.inter_sensor_gate_init)))
            for _ in range(cfg.inter_sensor_layers)
        ])

    # ---- sensor graph ------------------------------------------------------

    def adjacency(self):
        """Dense [C, C] sensor-dependency strength (row = receiver), pre top-k."""
        adj = None
        if self.use_learned:
            idx = torch.arange(self.C, device=self.emb_a.weight.device)
            a = torch.tanh(self.alpha * self.emb_a(idx))
            b = torch.tanh(self.alpha * self.emb_b(idx))
            adj = F.relu(torch.tanh(self.alpha * (a @ b.t() - b @ a.t())))
        if self.use_prior:
            adj = self.prior if adj is None else adj + self.prior
        return adj

    def top_k_edges(self):
        """
        Sparsify to each sensor's k strongest sources.
        Returns edge_index [2, C*k] (src=j -> dst=i) and edge_weight [C*k].
        Gradients reach the learned embeddings through edge_weight.
        """
        adj = self.adjacency().masked_fill(self.diag_mask, float("-inf"))
        vals, src = adj.topk(self.k, dim=-1)                       # [C, k]
        dst = torch.arange(self.C, device=adj.device).unsqueeze(1).expand_as(src)
        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        return edge_index, vals.reshape(-1)

    def _tile(self, edge_index, edge_weight, num_blocks):
        """
        Replicate the sensor graph into every (sample, scale) block.
        Spatial node layout is (b*3 + s)*C + c, so block id b*3+s starts at
        node offset (b*3+s)*C.
        """
        offsets = torch.arange(num_blocks, device=edge_index.device) * self.C
        ei = (edge_index.unsqueeze(1) + offsets.view(1, -1, 1)).reshape(2, -1)
        ew = edge_weight.repeat(num_blocks)
        return ei, ew

    # ---- forward -----------------------------------------------------------

    def forward(self, spatial_feats, batch_size):
        """
        spatial_feats: [batch * 3C, hidden]  (output of the spatial GNN)
        returns:       [batch * 3C, hidden]
        """
        edge_index, edge_weight = self.top_k_edges()
        ei, ew = self._tile(edge_index, edge_weight, batch_size * 3)
        edge_attr = ew.unsqueeze(-1)

        h = spatial_feats
        for gat, norm, gate in zip(self.gat_layers, self.norms, self.gates):
            h = norm(h + gate * gat(h, ei, edge_attr=edge_attr))
        return h


class FusionLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mode          = cfg.fusion_mode
        self.scale_match   = cfg.fusion_scale_match
        d                  = cfg.fusion_dim
        self.proj_temporal = nn.Linear(cfg.hidden_dim, d)
        self.proj_spatial  = nn.Linear(cfg.hidden_dim, d)
        self.norm          = nn.LayerNorm(d)
        if self.mode == "concat":
            self.mix = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        elif self.mode == "gated":
            self.gate = nn.Linear(2 * d, d)
        elif self.mode not in ("add", "temporal_only", "spatial_only"):
            raise ValueError(f"unknown fusion_mode {self.mode!r}")

    def forward(self, temporal_out, spatial_out, batch_size,
                num_temporal_nodes, num_channels, temporal_scale_labels):
        if self.mode == "temporal_only":
            return self.norm(self.proj_temporal(temporal_out))
        sp           = spatial_out.view(batch_size, 3, num_channels, -1)
        sp_per_scale = sp.mean(dim=2)                              # [B, 3, hidden]
        if not self.scale_match:                                   # ablation: no scale matching
            sp_per_scale = sp_per_scale.mean(dim=1, keepdim=True).expand(-1, 3, -1)
        T            = num_temporal_nodes
        batch_idx    = torch.arange(batch_size, device=spatial_out.device
                                    ).repeat_interleave(T)
        scale_idx    = temporal_scale_labels % 3
        sp_per_node  = sp_per_scale[batch_idx, scale_idx]         # [B*T, hidden]
        ps = self.proj_spatial(sp_per_node)
        if self.mode == "spatial_only":
            return self.norm(ps)
        pt = self.proj_temporal(temporal_out)
        if self.mode == "add":
            return self.norm(pt + ps)
        if self.mode == "concat":
            return self.norm(self.mix(torch.cat([pt, ps], dim=-1)))
        g = torch.sigmoid(self.gate(torch.cat([pt, ps], dim=-1)))   # gated
        return self.norm(g * pt + (1 - g) * ps)


# ===========================================================================
# CHANGE-A: Multi-head AnomalyScorer
# ===========================================================================

class AnomalyScorer(nn.Module):
    """
    Three lightweight O(C) reconstruction heads per temporal node:
      1. mean_head: reconstructs per-channel window mean  (same as v1)
      2. std_head:  reconstructs per-channel window std   (new — variance signal)
      3. corr_head: reconstructs per-channel window mean  (used for cosine error)

    All targets are shape [batch * T_nodes, C] — no expansion ratio problem.

    Anomaly score per node = weighted combination of the three per-node errors.
    Scale weights (fine < mid < coarse) remain as in v1.
    """
    SCALE_WEIGHTS = {0: 0.2, 1: 0.3, 2: 0.5}

    def __init__(self, cfg):
        super().__init__()
        self.cfg  = cfg
        C         = cfg.num_features
        h         = cfg.reconstruction_hidden
        d         = cfg.fusion_dim

        self.mean_head = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Linear(h, C)
        )
        self.std_head = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Linear(h, C), nn.Softplus()
        )
        # corr_head shares mean_head (cosine error uses same reconstruction)

        # Ablation: active heads, weights renormalised over the active ones
        heads = [h.strip() for h in cfg.score_heads.split(",") if h.strip()]
        bad = set(heads) - {"mean", "std", "corr"}
        if not heads or bad:
            raise ValueError(f"score_heads must be a subset of mean,std,corr; got {cfg.score_heads!r}")
        w = {"mean": cfg.mean_score_weight, "std": cfg.std_score_weight,
             "corr": cfg.corr_score_weight}
        tot = sum(w[h] for h in heads)
        self.head_w = {h: (w[h] / tot if h in heads else 0.0) for h in w}

    def forward(self, fused_feats, node_targets_mean, node_targets_std, scale_labels):
        """
        Args:
            fused_feats:       [batch * T_nodes, fusion_dim]
            node_targets_mean: [batch * T_nodes, C]
            node_targets_std:  [batch * T_nodes, C]
            scale_labels:      [batch * T_nodes]
        Returns:
            loss:           scalar
            per_node_error: [batch * T_nodes]
        """
        recon_mean = self.mean_head(fused_feats)          # [N, C]
        recon_std  = self.std_head(fused_feats)           # [N, C]

        # MSE errors — per-node scalar
        mean_err = F.mse_loss(recon_mean, node_targets_mean,
                               reduction="none").mean(dim=-1)          # [N]
        std_err  = F.mse_loss(recon_std,  node_targets_std,
                               reduction="none").mean(dim=-1)          # [N]

        # Cosine (pattern) error: 1 - cosine_similarity
        eps         = 1e-8
        dot         = (recon_mean * node_targets_mean).sum(dim=-1)
        norm_r      = recon_mean.norm(dim=-1).clamp(min=eps)
        norm_t      = node_targets_mean.norm(dim=-1).clamp(min=eps)
        corr_err    = 1.0 - dot / (norm_r * norm_t)                    # [N]

        
        per_node_error = (self.head_w["mean"] * mean_err
                        + self.head_w["std"]  * std_err
                        + self.head_w["corr"] * corr_err)

        scale_weights = torch.tensor(
            [self.SCALE_WEIGHTS[s.item()] for s in scale_labels],
            dtype=torch.float32, device=fused_feats.device,
        )
        loss = (per_node_error * scale_weights).mean()
        return loss, per_node_error


# ===========================================================================
# FULL MODEL
# ===========================================================================

class SpatioTemporalScaleGNN(nn.Module):
    """
    v3: Architecture unchanged from v1 (gives AUROC=0.905).
    Only the anomaly scorer and loss function are updated (CHANGE-A/B).

    Optional: InterSensorAttention (cfg.use_inter_sensor_attn) adds a GATv2
    layer over dependencies between different sensors inside each scale, applied
    to the spatial stream right after the cross-scale spatial GNN. Set
    use_inter_sensor_attn=False to recover the previous model exactly.
    """

    def __init__(self, cfg=None, sensor_prior=None):
        """
        sensor_prior: optional [C, C] |correlation| matrix from
                      build_sensor_prior(train_data); required when
                      cfg.inter_sensor_mode is "correlation" or "hybrid".
        """
        super().__init__()
        self.cfg = cfg or ScaleGNNConfig()
        if not self.cfg.multiscale_encoder:
            # Ablation: single-scale encoder. All three levels use the finest
            # span, so every node sees the same temporal resolution; graph,
            # fusion and scorer stay structurally unchanged.
            self.cfg = self.cfg._replace(mid_span=self.cfg.fine_span,
                                         coarse_span=self.cfg.fine_span)
        C = self.cfg.num_features

        self.scale_meta         = build_scale_metadata(self.cfg)
        t_edge_index            = build_temporal_edge_index(self.scale_meta)
        t_scale_labels          = build_temporal_scale_labels(self.scale_meta)
        self.num_temporal_nodes = sum(len(v) for v in self.scale_meta.values())
        self.register_buffer("t_edge_index",   t_edge_index)
        self.register_buffer("t_scale_labels", t_scale_labels)

        s_edge_index, s_scale_labels = build_spatial_graph(C)
        self.num_spatial_nodes = 3 * C
        self.register_buffer("s_edge_index",   s_edge_index)
        self.register_buffer("s_scale_labels", s_scale_labels)

        self.encoder           = MultiScaleEncoder(self.cfg)
        self.spatial_assembler = SpatialNodeAssembler(self.cfg)
        # Edge weighting per graph: "dynamic" | "static" | "none" (ablation)
        er, where = self.cfg.edge_refinement, self.cfg.edge_refine_where
        if er not in ("dynamic", "static", "none") or where not in ("both", "temporal", "spatial"):
            raise ValueError(f"bad edge_refinement={er!r} / edge_refine_where={where!r}")
        self.t_edge_mode = (er if where in ("both", "temporal") else "none") \
            if self.cfg.use_temporal_gnn else "none"
        self.s_edge_mode = (er if where in ("both", "spatial") else "none") \
            if self.cfg.use_spatial_gnn else "none"
        # Creation order kept as in the original model (t-refiner, t-GNN,
        # s-refiner, s-GNN) so the full model initialises identically per seed.
        self.t_edge_refiner = self._make_edge_module(self.t_edge_mode, t_edge_index.shape[1])
        self.t_gnn = ScaleAwareGNN(self.cfg) if self.cfg.use_temporal_gnn else None
        self.s_edge_refiner = self._make_edge_module(self.s_edge_mode, s_edge_index.shape[1])
        self.s_gnn = ScaleAwareGNN(self.cfg) if self.cfg.use_spatial_gnn else None
        # Inter-sensor attention runs after the cross-scale spatial GNN.
        self.inter_sensor = None
        if self.cfg.use_inter_sensor_attn and C >= 2:
            self.inter_sensor = InterSensorAttention(self.cfg, sensor_prior)
        self.fusion            = FusionLayer(self.cfg)
        self.scorer            = AnomalyScorer(self.cfg)

    def _make_edge_module(self, mode, num_edges):
        if mode == "dynamic":
            return DynamicEdgeRefinement(self.cfg.hidden_dim, self.cfg.edge_temperature)
        if mode == "static":
            return StaticEdgeWeights(num_edges)
        return None

    def _edge_weights(self, module, mode, feats, edges, batch_size):
        if mode == "dynamic":
            return module(feats, edges)
        if mode == "static":
            return module(batch_size)
        return torch.ones(edges.shape[1], device=feats.device)

    def _batch_temporal_graph(self, batch_size):
        n = self.num_temporal_nodes
        edges  = torch.cat([self.t_edge_index + i * n for i in range(batch_size)], dim=1)
        labels = self.t_scale_labels.repeat(batch_size)
        return edges, labels

    def _batch_spatial_graph(self, batch_size):
        n = self.num_spatial_nodes
        edges  = torch.cat([self.s_edge_index + i * n for i in range(batch_size)], dim=1)
        labels = self.s_scale_labels.repeat(batch_size)
        return edges, labels

    def _node_targets(self, x):
        """
        O(C) targets per node — same expansion ratio as v1.
        mean_targets: [batch * T_nodes, C]
        std_targets:  [batch * T_nodes, C]
        """
        batch_size = x.shape[0]
        mean_list, std_list = [], []
        for scale_name in ["fine", "mid", "coarse"]:
            for _, start, end in self.scale_meta[scale_name]:
                seg = x[:, start:end, :]            # [batch, span, C]
                mean_list.append(seg.mean(dim=1))   # [batch, C]
                std_list.append(seg.std(dim=1))     # [batch, C]

        # All nodes have the same target width C → safe to stack
        stacked_mean = torch.stack(mean_list, dim=0)   # [T_nodes, batch, C]
        stacked_std  = torch.stack(std_list,  dim=0)

        mean_out = stacked_mean.permute(1, 0, 2).reshape(
            batch_size * self.num_temporal_nodes, -1)
        std_out  = stacked_std.permute(1, 0, 2).reshape(
            batch_size * self.num_temporal_nodes, -1)
        return mean_out, std_out

    def forward(self, x):
        batch_size = x.shape[0]
        C = self.cfg.num_features

        temporal_node_feats, spatial_scale_feats = self.encoder(x, self.scale_meta)
        spatial_node_feats = self.spatial_assembler(spatial_scale_feats)

        bt_edges, bt_labels = self._batch_temporal_graph(batch_size)
        bs_edges, bs_labels = self._batch_spatial_graph(batch_size)

        fmode = self.cfg.fusion_mode
        # Temporal stream (skipped entirely if the fusion does not use it)
        temporal_out = temporal_node_feats
        if self.t_gnn is not None and fmode != "spatial_only":
            t_edge_weights = self._edge_weights(self.t_edge_refiner, self.t_edge_mode,
                                                temporal_node_feats, bt_edges, batch_size)
            temporal_out = self.t_gnn(temporal_node_feats, bt_edges, t_edge_weights, bt_labels)

        # Spatial stream (skipped entirely if the fusion does not use it)
        spatial_out = spatial_node_feats
        if fmode != "temporal_only":
            if self.s_gnn is not None:
                s_edge_weights = self._edge_weights(self.s_edge_refiner, self.s_edge_mode,
                                                    spatial_node_feats, bs_edges, batch_size)
                spatial_out = self.s_gnn(spatial_node_feats, bs_edges, s_edge_weights, bs_labels)
            if self.inter_sensor is not None:
                spatial_out = self.inter_sensor(spatial_out, batch_size)

        fused = self.fusion(temporal_out, spatial_out,
                            batch_size, self.num_temporal_nodes, C, bt_labels)

        mean_targets, std_targets = self._node_targets(x)
        loss, node_scores_flat = self.scorer(
            fused, mean_targets, std_targets, bt_labels
        )

        node_scores   = node_scores_flat.view(batch_size, self.num_temporal_nodes)
        anomaly_score = node_scores.max(dim=-1).values
        return loss, anomaly_score, node_scores


# ===========================================================================
# TRAINING
# ===========================================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss, _, _ = model(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ===========================================================================
# EVALUATION + CHANGE-C: Robust threshold calibration
# ===========================================================================

@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for x, y in loader:
        _, scores, _ = model(x.to(device))
        all_scores.append(scores.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def calibrate_threshold(scores, labels, train_scores=None,
                        percentile=99.0, min_precision=0.20, fbeta=0.5):
    """
    CHANGE-C: Precision-guarded threshold calibration.

    Key insight: when the score gap between normal and anomaly windows is
    small (as with reconstruction-based models), the raw PR curve F1-optimal
    threshold will always sit near zero (maximising recall).  We need to
    explicitly constrain precision to get a useful threshold.

    Five strategies are evaluated:
      A. F1-optimal on full PR curve           (v1 baseline — tends to recall=1)
      B. Fβ-optimal, β=0.5                    (precision-weighted F score)
      C. Best F1 subject to P ≥ min_precision  (direct precision floor)
      D. p99 of normal training scores         (anchored to normal distribution)
      E. Mean of val anomaly scores            (score distribution midpoint)

    Winner = highest val F1 among strategies with P ≥ min_precision.
    If none meet the floor, winner = strategy D (training percentile),
    which is the most reliable when val anomaly rate is low.
    """
    if labels.sum() == 0:
        print("  WARNING: no anomalies in val — using percentile threshold.")
        return float(np.percentile(train_scores, percentile)) \
            if train_scores is not None else float(np.median(scores))

    # Diagnostics: are anomaly scores actually higher than normal scores?
    anom_scores   = scores[labels == 1]
    normal_scores = scores[labels == 0]
    print(f"  Score gap — normal: mean={normal_scores.mean():.4f} std={normal_scores.std():.4f}  "
          f"| anomaly: mean={anom_scores.mean():.4f} std={anom_scores.std():.4f}  "
          f"| separation={anom_scores.mean() - normal_scores.mean():.4f}")

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(labels, scores)
    prec_arr = prec_arr[:-1]   # drop sentinel
    rec_arr  = rec_arr[:-1]

    def _f1(p, r):
        d = p + r; d[d == 0] = 1e-8
        return 2 * p * r / d

    def _fb(p, r, b):
        d = b**2 * p + r; d[d == 0] = 1e-8
        return (1 + b**2) * p * r / d

    f1_curve = _f1(prec_arr.copy(), rec_arr.copy())
    fb_curve = _fb(prec_arr.copy(), rec_arr.copy(), fbeta)

    # Strategy A — F1-optimal
    i_a      = int(np.argmax(f1_curve))
    thresh_a = float(thresh_arr[i_a])
    p_a, f_a = float(prec_arr[i_a]), float(f1_curve[i_a])

    # Strategy B — Fβ-optimal
    i_b      = int(np.argmax(fb_curve))
    thresh_b = float(thresh_arr[i_b])
    p_b      = float(prec_arr[i_b])
    f_b      = float(_f1(prec_arr[i_b:i_b+1], rec_arr[i_b:i_b+1])[0])

    # Strategy C — best F1 with precision floor
    valid = prec_arr >= min_precision
    if valid.any():
        masked     = np.where(valid, f1_curve, -1.0)
        i_c        = int(np.argmax(masked))
        thresh_c   = float(thresh_arr[i_c])
        p_c, f_c   = float(prec_arr[i_c]), float(f1_curve[i_c])
    else:
        # floor unreachable — slide to highest-precision point on the curve
        i_c        = int(np.argmax(prec_arr))
        thresh_c   = float(thresh_arr[i_c])
        p_c, f_c   = float(prec_arr[i_c]), float(f1_curve[i_c])
        print(f"  WARNING: min_precision={min_precision:.2f} unreachable; "
              f"best achievable P={p_c:.4f} at strategy C threshold")

    # Strategy D — percentile of normal training scores
    if train_scores is not None:
        thresh_d = float(np.percentile(train_scores, percentile))
    else:
        thresh_d = thresh_c   # safe fallback
    preds_d  = (scores >= thresh_d).astype(int)
    p_d = float(precision_score(labels, preds_d, zero_division=0))
    f_d = float(f1_score(labels, preds_d, zero_division=0))

    # Strategy E — midpoint between normal-p99 and anomaly-mean
    # Splits the score distributions at a data-driven midpoint
    if train_scores is not None:
        p99_normal = float(np.percentile(train_scores, 99))
        anom_mean  = float(np.percentile(scores[labels == 1], 50))
        thresh_e   = (p99_normal + anom_mean) / 2.0
    else:
        thresh_e = thresh_c
    preds_e  = (scores >= thresh_e).astype(int)
    p_e = float(precision_score(labels, preds_e, zero_division=0))
    f_e = float(f1_score(labels, preds_e, zero_division=0))

    candidates = {
        "A (F1-optimal)":      (thresh_a, p_a, f_a),
        "B (Fβ-optimal)":      (thresh_b, p_b, f_b),
        "C (precision floor)": (thresh_c, p_c, f_c),
        "D (p99 train)":       (thresh_d, p_d, f_d),
        "E (distribution mid)":(thresh_e, p_e, f_e),
    }

    # Winner = best F1 among those meeting precision floor
    meets_floor = {k: v for k, v in candidates.items() if v[1] >= min_precision}
    if meets_floor:
        winner = max(meets_floor, key=lambda k: meets_floor[k][2])
    else:
        # No strategy meets floor — use D (training percentile) as safe default
        winner = "D (p99 train)"
        print(f"  WARNING: no strategy meets precision floor {min_precision:.2f} "
              f"— defaulting to training percentile threshold")

    for name, (t, p, f) in candidates.items():
        mark = " ← WINNER" if name == winner else ""
        print(f"  {name}: threshold={t:.6f}  P={p:.4f}  F1={f:.4f}{mark}")

    return candidates[winner][0]


def _point_adjust(scores, labels, threshold):
    """PA protocol: credit whole anomaly segment if any window detected."""
    preds    = (scores >= threshold).astype(int)
    adjusted = preds.copy()
    in_seg   = False
    seg_start = 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_seg:
            in_seg, seg_start = True, i
        elif labels[i] == 0 and in_seg:
            if preds[seg_start:i].any():
                adjusted[seg_start:i] = 1
            in_seg = False
    if in_seg and preds[seg_start:].any():
        adjusted[seg_start:] = 1
    return adjusted


def _get_events(binary_arr):
    """Maximal runs of 1s in a 1D binary array, as inclusive (start, end) index pairs."""
    events, n, i = [], len(binary_arr), 0
    while i < n:
        if binary_arr[i] == 1:
            j = i
            while j < n and binary_arr[j] == 1:
                j += 1
            events.append((i, j - 1))
            i = j
        else:
            i += 1
    return events


def _dist_to_interval(x, s, e):
    if x < s: return s - x
    if x > e: return x - e
    return 0


def affiliation_f1(preds, labels):
    """
    Affiliation-based precision/recall/F1, approximating Huet et al.,
    "Local Evaluation of Time Series Anomaly Detection" (KDD 2022).

    The official metric works in continuous time and computes closed-form
    integrals over each ground-truth event's "zone of influence". This is a
    discretised re-implementation over the same per-window index array used
    everywhere else in this pipeline (raw predictions, before point-adjust) —
    it is meant to be reasonably faithful, not bit-for-bit identical to the
    reference `affiliation-metrics` package. Validate against the official
    package before quoting these numbers in a paper.

    Idea: split the timeline into one zone per ground-truth event, using the
    midpoints between consecutive events as boundaries. Within a zone, a
    predicted point is scored by its distance to the true event (precision)
    and a true-event point is scored by its distance to the nearest predicted
    point in that zone (recall); each score is normalised by the distance to
    the zone's own boundary, so a detection anywhere inside the zone gets
    partial credit even if it does not overlap the event exactly. Crucially,
    precision and recall are averaged PER EVENT, not per point — unlike raw
    or point-adjusted F1, one long anomaly segment cannot drown out several
    short ones, and a single-point detection cannot trivially inflate PA
    recall to 1.0 the way it does for point-adjust (see evaluate()'s PA
    protocol, which credits a whole segment from one hit).

    Returns (precision, recall, f1) as floats; (nan, nan, nan) if there are
    no ground-truth anomalies at all.
    """
    labels = np.asarray(labels).astype(int)
    preds  = np.asarray(preds).astype(int)
    n = len(labels)

    gt_events   = _get_events(labels)
    pred_events = _get_events(preds)
    if not gt_events:
        return float("nan"), float("nan"), float("nan")
    if not pred_events:
        return 0.0, 0.0, 0.0

    pred_idx = np.flatnonzero(preds)

    bounds = [0.0]
    for (_, e0), (s1, _) in zip(gt_events[:-1], gt_events[1:]):
        bounds.append((e0 + s1) / 2.0)
    bounds.append(float(n))

    precisions, recalls = [], []
    for k, (s, e) in enumerate(gt_events):
        zone_lo_i = min(int(np.ceil(bounds[k])), s)
        zone_hi_i = max(int(np.floor(bounds[k + 1])) - 1, e)
        zone_pred = pred_idx[(pred_idx >= zone_lo_i) & (pred_idx <= zone_hi_i)]

        # Recall: every point in the true event, scored by distance to the
        # nearest predicted point within this event's zone.
        rec_scores = []
        for x in range(s, e + 1):
            if len(zone_pred) == 0:
                rec_scores.append(0.0)
                continue
            d    = float(np.min(np.abs(zone_pred - x)))
            norm = max(x - zone_lo_i, zone_hi_i - x, 1)
            rec_scores.append(max(0.0, 1.0 - d / norm))
        recalls.append(float(np.mean(rec_scores)))

        # Precision: every predicted point in this zone, scored by distance
        # to the true event interval itself. An event with nothing predicted
        # anywhere in its zone contributes no precision evidence (skipped),
        # matching common affiliation-metric implementations.
        if len(zone_pred) == 0:
            continue
        prec_scores = []
        for x in zone_pred:
            d    = _dist_to_interval(int(x), s, e)
            norm = max(x - zone_lo_i, zone_hi_i - x, 1)
            prec_scores.append(max(0.0, 1.0 - d / norm))
        precisions.append(float(np.mean(prec_scores)))

    recall    = float(np.mean(recalls))    if recalls    else 0.0
    precision = float(np.mean(precisions)) if precisions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(scores, labels, threshold, split_name="test", point_adjust=True):
    raw_preds = (scores >= threshold).astype(int)
    pa_preds  = _point_adjust(scores, labels, threshold) if point_adjust else raw_preds

    def _m(preds, tag):
        return {
            f"{tag}_precision": float(precision_score(labels, preds, zero_division=0)),
            f"{tag}_recall":    float(recall_score(labels, preds, zero_division=0)),
            f"{tag}_f1":        float(f1_score(labels, preds, zero_division=0)),
        }

    raw_m = _m(raw_preds, "raw")
    pa_m  = _m(pa_preds,  "pa")
    auroc = float(roc_auc_score(labels, scores)) if labels.sum() > 0 else float("nan")

    aff_p, aff_r, aff_f1 = affiliation_f1(raw_preds, labels)
    aff_m = {"aff_precision": aff_p, "aff_recall": aff_r, "aff_f1": aff_f1}

    metrics = {**raw_m, **pa_m, **aff_m, "auroc": auroc}

    print(f"  [{split_name}] Raw  — "
          f"P={raw_m['raw_precision']:.4f}  R={raw_m['raw_recall']:.4f}  "
          f"F1={raw_m['raw_f1']:.4f}")
    print(f"  [{split_name}] PA   — "
          f"P={pa_m['pa_precision']:.4f}  R={pa_m['pa_recall']:.4f}  "
          f"F1={pa_m['pa_f1']:.4f}  AUROC={auroc:.4f}")
    print(f"  [{split_name}] Aff  — "
          f"P={aff_p:.4f}  R={aff_r:.4f}  F1={aff_f1:.4f}")
    return metrics


# ===========================================================================
# RESULT LOGGER
# ===========================================================================

class ResultLogger:
    RESULT_DIR = Path("RESULT")

    def __init__(self, dataset_name, cfg):
        self.dataset_name = dataset_name
        self.cfg          = cfg
        self.run_start    = datetime.now()
        self.epoch_rows   = []
        self.RESULT_DIR.mkdir(parents=True, exist_ok=True)

        ts             = self.run_start.strftime("%Y%m%d_%H%M%S")
        stem           = f"{dataset_name}_{ts}"
        self.log_path  = self.RESULT_DIR / f"{stem}.log"
        self.json_path = self.RESULT_DIR / f"{stem}.json"

        self.logger = logging.getLogger(stem)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        for h in [logging.FileHandler(self.log_path, "w", "utf-8"),
                  logging.StreamHandler()]:
            h.setFormatter(fmt)
            self.logger.addHandler(h)

        self.logger.info("=" * 60)
        self.logger.info(f"Run       : {stem}")
        self.logger.info(f"Dataset   : {dataset_name}")
        self.logger.info(f"Config    : {dict(cfg._asdict())}")
        self.logger.info("=" * 60)

    def log_epoch(self, epoch, train_loss, val_loss):
        row = {"epoch": epoch,
               "train_loss": round(train_loss, 6),
               "val_loss":   round(val_loss, 6)}
        self.epoch_rows.append(row)
        self.logger.info(
            f"Epoch {epoch:4d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    def log(self, msg):
        self.logger.info(msg)

    def save(self, threshold, metrics, cfg):
        finished = datetime.now()
        dur      = (finished - self.run_start).total_seconds()
        result   = {
            "run":      {"dataset": self.dataset_name,
                         "started_at":  self.run_start.isoformat(timespec="seconds"),
                         "finished_at": finished.isoformat(timespec="seconds"),
                         "duration_s":  round(dur, 1)},
            "config":   dict(cfg._asdict()),
            "training": self.epoch_rows,
            "threshold": round(threshold, 6),
            "metrics":  {k: round(float(v), 6) for k, v in metrics.items()},
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        self.logger.info("=" * 60)
        self.logger.info(f"Threshold : {threshold:.6f}")
        self.logger.info(f"Metrics   : {result['metrics']}")
        self.logger.info(f"JSON saved: {self.json_path}")
        self.logger.info(f"Log saved : {self.log_path}")
        self.logger.info(f"Duration  : {dur:.1f}s")
        self.logger.info("=" * 60)


# ===========================================================================
# END-TO-END PIPELINE
# ===========================================================================

def run_pipeline(train_data, train_labels, val_data, val_labels,
                 test_data, test_labels, cfg=None, dataset_name="dataset",
                 seed=None):
    """
    Full training + evaluation pipeline.

    Args:
        train_data / train_labels : (T_train, C) / (T_train,) — all normal
        val_data   / val_labels   : (T_val, C)   / (T_val,)
        test_data  / test_labels  : (T_test, C)  / (T_test,)
        cfg                       : ScaleGNNConfig (num_features inferred)
        dataset_name              : used in output filenames
        seed                      : if given, all RNGs are seeded with it
                                    before the model is built (see set_seed)
    Returns:
        model, calibrated_threshold, test_metrics
    """
    C = train_data.shape[1]
    assert val_data.shape[1]  == C
    assert test_data.shape[1] == C

    if seed is not None:
        set_seed(seed)

    cfg    = (cfg or ScaleGNNConfig())._replace(num_features=C)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl     = ResultLogger(dataset_name=dataset_name, cfg=cfg)
    rl.log(f"Device         : {device}")
    if device.type == "cuda":
        rl.log(f"GPU            : {torch.cuda.get_device_name(device)}")
    rl.log(f"Seed           : {seed}")
    rl.log(f"Channels (C)   : {C}  (inferred from data shape)")
    rl.log(f"Train shape    : {train_data.shape}  anomaly_rate={train_labels.mean():.2%}")
    rl.log(f"Val   shape    : {val_data.shape}    anomaly_rate={val_labels.mean():.2%}")
    rl.log(f"Test  shape    : {test_data.shape}   anomaly_rate={test_labels.mean():.2%}")

    train_loader, val_loader, test_loader, _ = prepare_splits(
        train_data, train_labels, val_data, val_labels,
        test_data, test_labels, cfg
    )
    rl.log(f"Train windows  : {len(train_loader.dataset)}  (all normal)")
    rl.log(f"Val   windows  : {len(val_loader.dataset)}")
    rl.log(f"Test  windows  : {len(test_loader.dataset)}")

    # Inter-sensor correlation prior — fit on train_data only (no leakage).
    sensor_prior = None
    if cfg.use_inter_sensor_attn and cfg.inter_sensor_mode in ("correlation", "hybrid"):
        sensor_prior = build_sensor_prior(train_data)
    if cfg.use_inter_sensor_attn:
        rl.log(f"Inter-sensor attention : mode={cfg.inter_sensor_mode}  "
               f"top_k={cfg.inter_sensor_top_k}  layers={cfg.inter_sensor_layers}  "
               f"gate_init={cfg.inter_sensor_gate_init}")
    else:
        rl.log("Inter-sensor attention : disabled (use_inter_sensor_attn=False)")

    model    = SpatioTemporalScaleGNN(cfg, sensor_prior=sensor_prior).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rl.log(f"Trainable parameters : {n_params:,}")
    n_train_windows = len(train_loader.dataset)
    rl.log(f"Temporal graph nodes : {model.num_temporal_nodes}")
    rl.log(f"Spatial  graph nodes : {model.num_spatial_nodes}  (= 3 × {C})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs)

    rl.log(f"Training for {cfg.num_epochs} epochs...")
    best_val_loss = float("inf")
    best_state    = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                l, _, _ = model(x.to(device))
                val_loss += l.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}

        rl.log_epoch(epoch, train_loss, val_loss)

    train_time_s = time.time() - train_start
    peak_mem_train_gb = (torch.cuda.max_memory_allocated(device) / 1e9
                         if device.type == "cuda" else 0.0)
    rl.log(f"Training time  : {train_time_s:.2f}s  "
           f"({train_time_s / cfg.num_epochs:.3f}s/epoch)  "
           f"peak_mem={peak_mem_train_gb:.3f}GB")

    model.load_state_dict(best_state)
    rl.log("Best checkpoint restored.")

    # Collect train scores (all normal) for percentile threshold
    rl.log("Collecting training scores...")
    train_scores, _ = collect_scores(model, train_loader, device)
    rl.log(f"  Train score — mean={train_scores.mean():.6f}  "
           f"std={train_scores.std():.6f}  "
           f"p99={np.percentile(train_scores, 99):.6f}  "
           f"p99.5={np.percentile(train_scores, 99.5):.6f}  "
           f"max={train_scores.max():.6f}")

    rl.log("Calibrating threshold on validation set...")
    val_scores, val_lbls = collect_scores(model, val_loader, device)
    rl.log(f"  Val score   — mean={val_scores.mean():.6f}  "
           f"std={val_scores.std():.6f}  "
           f"max={val_scores.max():.6f}")

    threshold = calibrate_threshold(
        val_scores, val_lbls,
        train_scores=train_scores,
        percentile=cfg.threshold_percentile,
        min_precision=cfg.min_precision,
        fbeta=cfg.fbeta,
    )
    rl.log(f"  Calibrated threshold: {threshold:.6f}")

    rl.log("Test set evaluation:")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)   # isolate inference-only memory
    inference_start = time.time()
    test_scores, test_lbls = collect_scores(model, test_loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time_s = time.time() - inference_start
    peak_mem_inference_gb = (torch.cuda.max_memory_allocated(device) / 1e9
                             if device.type == "cuda" else 0.0)
    n_test_windows = len(test_loader.dataset)
    inference_ms_per_window = inference_time_s / max(n_test_windows, 1) * 1000

    rl.log(f"Inference time : {inference_time_s:.4f}s total  "
           f"({inference_ms_per_window:.4f}ms/window over {n_test_windows} windows)  "
           f"peak_mem={peak_mem_inference_gb:.3f}GB")

    metrics = evaluate(test_scores, test_lbls, threshold,
                       split_name="test", point_adjust=True)
    metrics.update({
        "train_time_s":             round(train_time_s, 4),
        "train_time_s_per_epoch":   round(train_time_s / cfg.num_epochs, 4),
        "peak_mem_train_gb":        round(peak_mem_train_gb, 4),
        "inference_time_s":         round(inference_time_s, 4),
        "inference_ms_per_window":  round(inference_ms_per_window, 4),
        "peak_mem_inference_gb":    round(peak_mem_inference_gb, 4),
        "n_params":                 n_params,
        "n_train_windows":          n_train_windows,
        "n_test_windows":           n_test_windows,
        "seed":                     -1 if seed is None else int(seed),
    })

    rl.save(threshold, metrics, cfg)
    return model, threshold, metrics


# ===========================================================================
# EXAMPLE / SANITY CHECK
# ===========================================================================

if __name__ == "__main__":
    np.random.seed(42)

    T_train, T_val, T_test, C = 21000, 3000, 3131, 38

    train_data   = np.random.randn(T_train, C).astype(np.float32)
    train_labels = np.zeros(T_train, dtype=int)
    val_data     = np.random.randn(T_val, C).astype(np.float32)
    val_labels   = (np.random.rand(T_val) < 0.05).astype(int)
    test_data    = np.random.randn(T_test, C).astype(np.float32)
    test_labels  = (np.random.rand(T_test) < 0.05).astype(int)

    cfg = ScaleGNNConfig(
        window_size=12, stride=1,
        fine_span=3, mid_span=6, coarse_span=12,
        hidden_dim=64, gat_heads=4, num_gnn_layers=2,
        reconstruction_hidden=64, fusion_dim=64,
        use_inter_sensor_attn=True,      # inter-sensor graph attention
        inter_sensor_mode="hybrid",      # "learned" | "correlation" | "hybrid"
        inter_sensor_top_k=5,
        mean_score_weight=0.5,   # CHANGE-A
        std_score_weight=0.3,
        corr_score_weight=0.2,
        batch_size=256,
        num_epochs=30,
        learning_rate=1e-3,
        min_precision=0.20,      # CHANGE-C
        fbeta=0.5,
        threshold_percentile=99.0,
    )

    model, threshold, metrics = run_pipeline(
        train_data, train_labels,
        val_data,   val_labels,
        test_data,  test_labels,
        cfg=cfg, dataset_name="MSL",
    )
    print(f"\nFinal test PA-F1  : {metrics['pa_f1']:.4f}")
    print(f"Final test Aff-F1 : {metrics['aff_f1']:.4f}")
    print(f"Final test AUROC  : {metrics['auroc']:.4f}")
    print(f"Train time        : {metrics['train_time_s']:.2f}s "
          f"({metrics['train_time_s_per_epoch']:.3f}s/epoch, "
          f"peak_mem={metrics['peak_mem_train_gb']:.3f}GB)")
    print(f"Inference time    : {metrics['inference_time_s']:.4f}s "
          f"({metrics['inference_ms_per_window']:.4f}ms/window, "
          f"peak_mem={metrics['peak_mem_inference_gb']:.3f}GB)")
