"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/11/24
"""
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch_geometric.utils import negative_sampling
import torch_geometric as pyg
import torch.nn.functional as F

class ResidualAdd(torch.nn.Module):
    def forward(self, x, x_residual):
        return x + x_residual


class MaskedReconstructionLoss(_Loss):
    def __init__(self):
        super().__init__()

    # todo: handle train, val, and test data separately
    def forward(self, predicted_edge_label: Tensor, target_edge_label: Tensor, *args, **kwargs) -> Tensor:
        # target is the torch_geometric.data.Data object
        # input and target['edge_index'] are both adjacency matrices with complimentary masked region
        # the input of baseline model is already an adjacency matrix with continuous values [0, 1]
        # only need to convert the target to an adjacency matrix

        # Todo: assuming the adjacency matrix of input and target are the same in terms of node order (row and col)
        #   is it safe?
        loss = F.binary_cross_entropy_with_logits(predicted_edge_label, target_edge_label).mean()
        return loss


