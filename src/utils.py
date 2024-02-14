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

    def forward(self, input: Tensor, target: pyg.data.Data, *args, **kwargs) -> Tensor:
        # target is the torch_geometric.data.Data object
        # input and target['edge_index'] are both adjacency matrices with complimentary masked region
        # the input of baseline model is already an adjacency matrix with continuous values [0, 1]
        # only need to convert the target to an adjacency matrix

        pos_edge_index = target['edge_index']
        num_pos_edge = pos_edge_index.size(1)

        loss_edge_index = torch.cat((pos_edge_index, target.neg_edge_index), dim=-1)
        loss_target = torch.cat((torch.ones(num_pos_edge, dtype=torch.float, device=input.device),
                                 torch.zeros(target.neg_edge_index.size(1), dtype=torch.float, device=input.device)),
                                dim=-1)
        # use BCEWITHLOGITSLOSS to avoid numerical instability
        # Todo: assuming the adjacency matrix of input and target are the same in terms of node order (row and col)
        #   is it safe?
        loss = F.binary_cross_entropy_with_logits(input[loss_edge_index[0], loss_edge_index[1]], loss_target).mean()
        return loss


