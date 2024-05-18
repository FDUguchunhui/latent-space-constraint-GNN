"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/11/24
"""
import torch
from torch import Tensor
from torch_geometric.utils import negative_sampling
import torch_geometric as pyg
import torch.nn.functional as F

class ResidualAdd(torch.nn.Module):
    def forward(self, x, x_residual):
        return x + x_residual


class MaskedReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # todo: handle train, val, and test data separately
    def forward(self, input_edge_index: Tensor, target: pyg.data.Data, mask, *args, **kwargs) -> Tensor:
        # target is the torch_geometric.data.Data object
        # input and target['edge_index'] are both adjacency matrices with complimentary masked region
        # the input of baseline model is already an adjacency matrix with continuous values [0, 1]
        # only need to convert the target to an adjacency matrix

        target_edges = target.edge_index[:, target[mask].squeeze(-1)]
        target_labels = target.edge_label[target[mask]]
        input_edge_index = input_edge_index[target[mask]]
        # use BCEWITHLOGITSLOSS to avoid numerical instability
        # Todo: assuming the adjacency matrix of input and target are the same in terms of node order (row and col)
        #   is it safe?
        loss = F.binary_cross_entropy_with_logits(input_edge_index, target_labels).mean()
        return loss


