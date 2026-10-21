import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, input_tensor, adj_matrix=None):
        output = self.linear(input_tensor)
        if self.activation:
            output = self.activation(output)
        return output

class CrossScaleAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, adj=None):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

class CrossScaleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, fine_len, mid_len, coarse_len):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear_fine = nn.Linear(input_dim, hidden_dim)
        self.linear_mid = nn.Linear(input_dim, hidden_dim)
        self.linear_coarse = nn.Linear(input_dim, hidden_dim)

        self.graph_f2m = CrossScaleAttention(hidden_dim)
        self.graph_m2c = CrossScaleAttention(hidden_dim)
        self.graph_c2f = CrossScaleAttention(hidden_dim)
        self.graph_m2f = CrossScaleAttention(hidden_dim)
        self.graph_c2m = CrossScaleAttention(hidden_dim)
        self.graph_f2c = CrossScaleAttention(hidden_dim)

        self.gcn_fine = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn_mid = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn_coarse = GraphConvolution(hidden_dim, hidden_dim)

    def forward(self, h_fine, h_mid, h_coarse, graphs=None):
        h_fine_proj = F.relu(self.linear_fine(h_fine))
        h_mid_proj = F.relu(self.linear_mid(h_mid))
        h_coarse_proj = F.relu(self.linear_coarse(h_coarse))

        _h_fine = h_fine_proj.unsqueeze(1)
        _h_mid = h_mid_proj.unsqueeze(1)
        _h_coarse = h_coarse_proj.unsqueeze(1)

        f2m_output, _ = self.graph_f2m(_h_fine, _h_mid, _h_mid, adj=graphs)
        m2c_output, _ = self.graph_m2c(_h_mid, _h_coarse, _h_coarse, adj=graphs)
        c2f_output, _ = self.graph_c2f(_h_coarse, _h_fine, _h_fine, adj=graphs)

        m2f_output, _ = self.graph_m2f(_h_mid, _h_fine, _h_fine, adj=graphs)
        c2m_output, _ = self.graph_c2m(_h_coarse, _h_mid, _h_mid, adj=graphs)
        f2c_output, _ = self.graph_f2c(_h_fine, _h_coarse, _h_coarse, adj=graphs)

        return h_fine_proj, h_mid_proj, h_coarse_proj

class CrossScaleTemporalGCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, fine_len, mid_len, coarse_len, topk):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fine_len = fine_len
        self.mid_len = mid_len
        self.coarse_len = coarse_len
        self.topk = topk

        self.gcn = CrossScaleGCN(input_dim, hidden_dim, fine_len, mid_len, coarse_len)
        self.reconstruction_layer = nn.Linear(hidden_dim * 3, input_dim)

    def forward(self, h_fine, h_mid, h_coarse):
        h_fine_gcn, h_mid_gcn, h_coarse_gcn = self.gcn(h_fine, h_mid, h_coarse)

        combined_h = torch.cat((h_fine_gcn, h_mid_gcn, h_coarse_gcn), dim=-1)
        recon = self.reconstruction_layer(combined_h)
        return recon