"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/10/24
"""

import torch_geometric as pyg
from overrides import overrides
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.loader import DataLoader
import torch

# @functional_transform('mask_adjacency_matrix')
class MaskAdjacencyMatrix(BaseTransform):
    '''
    This transformation masks the adjacency matrix of torch_geometric dataset.
    '''

    @overrides
    def forward(self, data):
        adj_mat = pyg.utils.to_dense_adj(data.edge_index).squeeze()
        out_adj_mat = adj_mat.clone() # create a masked version of the adjacency matrix
        dim, _ = adj_mat.shape # get num of nodes

        # create a masked version of the adjacency matrix `adj` for input and output
        # the index `adj` is divided into 4 quadrants. All quadrants except the bottom-right are used as input and
        # the bottom-right quadrant is used as output.
        # For `adj`, when the value is masked, it is set to 0 to avoid information aggregation through masked edges.
        # It was tried to set it as 0.5 to denote the that the edge is neither 0 (non-existent) but the result is not
        # as expected.
        MASK_VALUE = 0
        # TODO: the rule about what region of adjacency matrix is masked is different from the mask in image since
        #   the adjacency matrix is symmetric and the image is not necessarily symmetric. Current implementation only
        #   masks the bottom-right quadrant of the adjacency matrix. But it should support more masking options later.
        #   nor 1 (existent)
        # mask quadrant 1 & 2
        out_adj_mat[:, dim // 2:] = MASK_VALUE
        # mask quadrant 3
        out_adj_mat[dim // 2:, : dim // 2] = MASK_VALUE
        out_edge_index, out_edge_weight = pyg.utils.dense_to_sparse(out_adj_mat)
        out = Data(data.x, edge_index=out_edge_index, edge_weight=out_edge_weight)

        # for input only mask the bottom-right quadrant
        inp_adj_mat = adj_mat.clone()
        # mask quadrant 4
        inp_adj_mat[dim // 2:, dim // 2:] = MASK_VALUE
        inp_edge_index, inp_edge_weight = pyg.utils.dense_to_sparse(inp_adj_mat)
        inp = Data(data.x, edge_index=inp_edge_index, edge_weight=inp_edge_weight)
        sample = {'input': inp, 'output': out}
        return sample

def get_data():
    # load data
    dataset = Planetoid(root='.', name='Cora', transform=MaskAdjacencyMatrix())
    # dataset = KarateClub(transform=MaskAdjacencyMatrix())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataset_sizes = len(dataset)
    return dataloader, dataset_sizes
