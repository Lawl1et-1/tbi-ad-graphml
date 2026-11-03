"""Graph-level GAT classifier for gene-expression graphs with a shared topology.

Every sample is the same gene graph; only node features (that sample's expression) differ.
Node input = [expression value, learned gene-identity embedding]. Without the embedding the
pooling step could not tell which gene a value came from.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, to_undirected


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_edge_index(edges, genes: list[str]) -> torch.Tensor:
    """edges: DataFrame with columns a, b (gene symbols). Returns undirected edge_index with self-loops."""
    idx = {g: i for i, g in enumerate(genes)}
    if len(edges):
        ei = torch.tensor([[idx[a] for a in edges.a], [idx[b] for b in edges.b]], dtype=torch.long)
        ei = to_undirected(ei, num_nodes=len(genes))
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
    return add_self_loops(ei, num_nodes=len(genes))[0]


def batch_graph(edge_index: torch.Tensor, n_nodes: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy one graph b times into one disconnected big graph."""
    offs = (torch.arange(b) * n_nodes).repeat_interleave(edge_index.shape[1])
    ei = edge_index.repeat(1, b) + offs
    batch = torch.arange(b).repeat_interleave(n_nodes)
    return ei, batch


class GeneGAT(nn.Module):
    def __init__(self, n_genes: int, emb_dim: int = 8, hidden: int = 16, heads: int = 4, dropout: float = 0.2,
                 readout: str = "pool"):
        """readout='pool': mean+max pooling then MLP (v1).
        readout='gene': each gene's final state -> scalar, weighted by a learned per-gene weight, summed.
        The gene readout keeps gene identity, so the graph-level score is a linear model on
        network-smoothed node states."""
        super().__init__()
        self.n_genes, self.readout = n_genes, readout
        self.emb = nn.Embedding(n_genes, emb_dim)
        self.inp = nn.Linear(1 + emb_dim, hidden)
        self.gat1 = GATConv(hidden, hidden, heads=heads, dropout=dropout, add_self_loops=False)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, add_self_loops=False)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.node_score = nn.Linear(hidden, 1)
        self.gene_w = nn.Parameter(torch.zeros(n_genes))
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = dropout

    def node_embed(self, x: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False):
        """x: (B, n_genes) expression. Returns node states (B*n_genes, hidden)."""
        B = x.shape[0]
        gid = torch.arange(self.n_genes, device=x.device).repeat(B)
        h = torch.cat([x.reshape(-1, 1), self.emb(gid)], dim=1)
        h = F.elu(self.inp(h))
        h = F.dropout(h, self.dropout, self.training)
        if return_attention:
            h, att1 = self.gat1(h, edge_index, return_attention_weights=True)
        else:
            h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, self.dropout, self.training)
        if return_attention:
            h, att2 = self.gat2(h, edge_index, return_attention_weights=True)
            return F.elu(h), (att1, att2)
        return F.elu(self.gat2(h, edge_index))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.node_embed(x, edge_index)
        if self.readout == "gene":
            s = self.node_score(h).view(x.shape[0], self.n_genes)  # (B, genes)
            return (s * self.gene_w).sum(1) / np.sqrt(self.n_genes) + self.bias
        g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.head(g).squeeze(-1)


class GATClassifier:
    """sklearn-style wrapper: fit(X, y, X_val, y_val) / predict_proba(X)."""

    def __init__(self, edge_index: torch.Tensor, n_genes: int, seed: int = 42, epochs: int = 60,
                 batch_size: int = 16, lr: float = 3e-3, weight_decay: float = 1e-4, patience: int = 10,
                 min_epochs: int = 0, **kw):
        self.edge_index, self.n_genes, self.seed = edge_index, n_genes, seed
        self.min_epochs = min_epochs
        self.epochs, self.bs, self.lr, self.wd, self.patience, self.kw = epochs, batch_size, lr, weight_decay, patience, kw
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _graph(self, b: int):
        if b not in self._cache:
            self._cache[b] = batch_graph(self.edge_index, self.n_genes, b)
        return self._cache[b]

    def _logits(self, X: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(0, len(X), self.bs):
            xb = X[i:i + self.bs]
            ei, batch = self._graph(len(xb))
            out.append(self.model(xb, ei, batch))
        return torch.cat(out)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        set_seed(self.seed)
        self.model = GeneGAT(self.n_genes, **self.kw)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        Xv, yv = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
        pos_w = torch.tensor((1 - y.mean()) / y.mean())
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, wait, self.history = np.inf, None, 0, []
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), self.bs):
                idx = perm[i:i + self.bs]
                ei, batch = self._graph(len(idx))
                loss = F.binary_cross_entropy_with_logits(self.model(X[idx], ei, batch), y[idx], pos_weight=pos_w)
                opt.zero_grad(); loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = F.binary_cross_entropy_with_logits(self._logits(Xv), yv, pos_weight=pos_w).item()
            self.history.append(vl)
            if ep < self.min_epochs:  # do not pick a model before it had time to learn
                best_state, best_ep = copy.deepcopy(self.model.state_dict()), ep
                continue
            if vl < best - 1e-4:
                best, best_state, wait, best_ep = vl, copy.deepcopy(self.model.state_dict()), 0, ep
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.best_epoch = int(best_ep)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            p = torch.sigmoid(self._logits(torch.tensor(X, dtype=torch.float32))).numpy()
        return np.column_stack([1 - p, p])
