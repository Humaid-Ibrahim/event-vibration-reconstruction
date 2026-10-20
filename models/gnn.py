"""
Graph Neural Network model for ROI-based event-to-vibration prediction.

Architecture:
    1. ROIs arranged in a spatial grid form a graph
    2. Per-node 1D CNN encoder extracts temporal embeddings
    3. GNN (GAT/GCN) propagates information between neighboring ROIs
    4. Global pooling + MLP decoder generates vibration output

Input: (B, C, T) where C = channels_per_roi * num_rois, T = 2048
Output: (B, T) vibration signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


def build_roi_graph(grid_shape, neighborhood='8'):
    """
    Build graph connectivity for ROI grid.

    Args:
        grid_shape: (rows, cols) ROI grid dimensions
        neighborhood: '4' for 4-connected, '8' for 8-connected

    Returns:
        edge_index: (2, num_edges) edge connectivity tensor
        num_nodes: total number of ROI nodes
    """
    rows, cols = grid_shape
    num_nodes = rows * cols

    def node_idx(r, c):
        return r * cols + c

    edges = []

    for r in range(rows):
        for c in range(cols):
            idx = node_idx(r, c)

            # Right neighbor
            if c + 1 < cols:
                edges.append([idx, node_idx(r, c + 1)])

            # Bottom neighbor
            if r + 1 < rows:
                edges.append([idx, node_idx(r + 1, c)])

            if neighborhood == '8':
                # Bottom-right diagonal
                if r + 1 < rows and c + 1 < cols:
                    edges.append([idx, node_idx(r + 1, c + 1)])

                # Bottom-left diagonal
                if r + 1 < rows and c - 1 >= 0:
                    edges.append([idx, node_idx(r + 1, c - 1)])

    # Make undirected
    edges_reversed = [[j, i] for i, j in edges]
    edges = edges + edges_reversed

    # Add self-loops
    self_loops = [[i, i] for i in range(num_nodes)]
    edges = edges + self_loops

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, num_nodes


class _TemporalROIEncoder(nn.Module):
    """Per-ROI temporal feature encoder using 1D CNN."""

    def __init__(self, channels_per_roi=2, embed_dim=64, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(channels_per_roi, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.projection(z)


class GNNModel(nn.Module):
    """GNN-based model for event-to-vibration prediction."""

    def __init__(
        self,
        grid_shape=(5, 5),
        channels_per_roi=1,
        output_size=2048,
        embed_dim=64,
        gnn_hidden_dim=128,
        num_gnn_layers=3,
        dropout=0.3,
        neighborhood='8',
        use_attention=True,
    ):
        super().__init__()

        self.grid_shape = grid_shape
        self.channels_per_roi = channels_per_roi
        self.num_rois = grid_shape[0] * grid_shape[1]
        self.embed_dim = embed_dim

        # Build graph structure
        self.edge_index, _ = build_roi_graph(grid_shape, neighborhood=neighborhood)
        self.register_buffer('graph_edge_index', self.edge_index)

        # Cache for batched edge indices
        self._batched_edge_cache = {}

        # Temporal encoder (shared across ROIs)
        self.roi_encoder = _TemporalROIEncoder(
            channels_per_roi=channels_per_roi,
            embed_dim=embed_dim,
            dropout=dropout
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        in_dim = embed_dim
        for i in range(num_gnn_layers):
            out_dim = gnn_hidden_dim if i < num_gnn_layers - 1 else embed_dim

            if use_attention:
                self.gnn_layers.append(
                    GATConv(in_dim, out_dim, heads=4, concat=False, dropout=dropout)
                )
            else:
                self.gnn_layers.append(GCNConv(in_dim, out_dim))

            self.gnn_norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        # Global pooling + decoder
        pooled_dim = embed_dim * 2  # mean + max pooling

        self.decoder = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),

            nn.Linear(1024, output_size),
        )

    def forward(self, x):
        B, C_total, T = x.shape

        # Reshape to (B, num_rois, channels_per_roi, T)
        x = x.view(B, self.num_rois, self.channels_per_roi, T)

        # Process all ROIs from all batches together
        x_flat = x.view(B * self.num_rois, self.channels_per_roi, T)

        # Encode all ROIs
        h = self.roi_encoder(x_flat)

        # Batch indices for pooling
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(self.num_rois)

        # Expand edge_index for batch (with caching)
        if B not in self._batched_edge_cache:
            edge_index_list = []
            for b in range(B):
                offset = b * self.num_rois
                edge_index_list.append(self.graph_edge_index + offset)
            batched_edge_index = torch.cat(edge_index_list, dim=1)
            self._batched_edge_cache[B] = batched_edge_index
        else:
            batched_edge_index = self._batched_edge_cache[B]

        # Apply GNN layers
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            h_new = gnn(h, batched_edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)

            if h_new.shape == h.shape:
                h = h_new + h  # residual
            else:
                h = h_new

        # Global pooling
        h_mean = global_mean_pool(h, batch_idx)

        # Manual max pooling
        h_max_list = []
        for b in range(B):
            mask = batch_idx == b
            h_max_list.append(h[mask].max(dim=0)[0])
        h_max = torch.stack(h_max_list, dim=0)

        h_global = torch.cat([h_mean, h_max], dim=1)

        return self.decoder(h_global)
