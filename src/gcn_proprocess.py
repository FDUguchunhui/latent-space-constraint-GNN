import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

# pyg implementation of GCNJaccard
class GCN_2layer(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0, threshold=0.01):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.threshold = threshold

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x


import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse

def dropedge_jaccard(edge_index, features, threshold):
    # Convert edge_index to dense adjacency matrix
    adj = to_dense_adj(edge_index).squeeze(0)

    # Compute similarity matrix using GCN_2layer similarity
    intersection = torch.mm(features, features.t())
    union = features.sum(dim=1).unsqueeze(1) + features.sum(dim=1).unsqueeze(0) - intersection
    similarity = intersection / union

    # Drop edges based on similarity threshold
    adj[similarity < threshold] = 0

    # Convert back to edge_index
    edge_index = dense_to_sparse(adj)[0]

    return edge_index

import torch

def truncatedSVD(data, k=50):
    """Truncated SVD on input data.

    Parameters
    ----------
    data :
        input matrix to be decomposed
    k : int
        number of singular values and vectors to compute.

    Returns
    -------
    numpy.array
        reconstructed matrix.
    """
    print('=== GCN-SVD: rank={} ==='.format(k))
    if sp.issparse(data):
        data = data.asfptype()
        U, S, V = sp.linalg.svds(data, k=k)
        print("rank_after = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
    else:
        U, S, V = np.linalg.svd(data)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
        print("rank_before = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
        print("rank_after = {}".format(len(diag_S.nonzero()[0])))

    return U @ diag_S @ V