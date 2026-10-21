# hierarchical_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError("Please install torch-geometric: pip install torch-geometric")


# -------------------------------------------------
# Graph Builder
# -------------------------------------------------
class HierarchicalGraph:
    def __init__(self, T_fine, k1, k2, device="cpu"):
        self.T_fine = T_fine
        self.k1 = k1
        self.k2 = k2
        self.device = device

    def build(self):
        fine_idx = torch.arange(self.T_fine, device=self.device)

        # Fine → Mid
        mid_idx = fine_idx // self.k1
        T_mid = mid_idx.max().item() + 1

        A_fine_mid = torch.zeros(self.T_fine, T_mid, device=self.device)
        A_fine_mid[torch.arange(self.T_fine), mid_idx] = 1

        # Mid → Coarse
        mid_nodes = torch.arange(T_mid, device=self.device)
        coarse_idx = mid_nodes // self.k2
        T_coarse = coarse_idx.max().item() + 1

        A_mid_coarse = torch.zeros(T_mid, T_coarse, device=self.device)
        A_mid_coarse[mid_nodes, coarse_idx] = 1

        return A_fine_mid, A_mid_coarse

    def build_edge_index(self):
        A_fine_mid, A_mid_coarse = self.build()

        T_fine, T_mid = A_fine_mid.shape
        _, T_coarse = A_mid_coarse.shape

        total_nodes = T_fine + T_mid + T_coarse

        fine_offset = 0
        mid_offset = T_fine
        coarse_offset = T_fine + T_mid

        edges = []

        # Fine ↔ Mid
        fine_idx, mid_idx = torch.nonzero(A_fine_mid, as_tuple=True)
        for f, m in zip(fine_idx, mid_idx):
            edges.append([f.item(), mid_offset + m.item()])
            edges.append([mid_offset + m.item(), f.item()])  # bidirectional

        # Mid ↔ Coarse
        mid_idx, coarse_idx = torch.nonzero(A_mid_coarse, as_tuple=True)
        for m, c in zip(mid_idx, coarse_idx):
            edges.append([mid_offset + m.item(), coarse_offset + c.item()])
            edges.append([coarse_offset + c.item(), mid_offset + m.item()])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return edge_index, total_nodes


# -------------------------------------------------
# GNN Model
# -------------------------------------------------
class HierarchicalGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        #print('In Dims : ---', in_dim)
        #print('Hidden Dims : ---', hidden_dim)
        #print('Out Dims : ---', out_dim)

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        #print('X in HeirarchicalGNN1 --- : ', x.shape)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        print('X in HeirarchicalGNN2 --- : ', x.shape)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        print('X in HeirarchicalGNN3 --- : ', x.shape)

        x = self.conv3(x, edge_index)

        print('X in HeirarchicalGNN4 --- : ', x.shape)

        return x


# -------------------------------------------------
# Demo / Training Example
# -------------------------------------------------
def demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Graph parameters ----
    T_fine = 8
    k1 = 2
    k2 = 2

    graph_builder = HierarchicalGraph(T_fine, k1, k2, device)
    edge_index, num_nodes = graph_builder.build_edge_index()

    # ---- Node features ----
    in_dim = 4
    x = torch.randn(num_nodes, in_dim, device=device)

    # ---- Dummy labels (node classification) ----
    out_dim = 2
    y = torch.randint(0, out_dim, (num_nodes,), device=device)

    # ---- Model ----
    model = HierarchicalGNN(in_dim, hidden_dim=16, out_dim=out_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ---- Training loop ----
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()

        out = model(x, edge_index)

        loss = F.cross_entropy(out, y)
        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("\nFinal node embeddings:")
    print(out)