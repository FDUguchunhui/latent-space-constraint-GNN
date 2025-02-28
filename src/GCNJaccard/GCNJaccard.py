import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

# pyg implementation of GCNJaccard
class Jaccard(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0, threshold=0.01):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.threshold = threshold

    def forward(self, data):

        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x
