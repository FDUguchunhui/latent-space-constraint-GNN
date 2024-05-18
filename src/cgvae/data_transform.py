"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/10/24
"""
from typing import Any

import torch_geometric as pyg
import torch_geometric.seed
from overrides import overrides
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.transforms import BaseTransform, ToUndirected, AddSelfLoops, RandomLinkSplit
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.loader import DataLoader
import torch
import numpy as np

# @functional_transform('mask_adjacency_matrix')
class MaskAdjacencyMatrix(BaseTransform):
    '''
    This transformation masks the adjacency matrix of torch_geometric dataset.
    '''

    def __init__(self, neg_edge_ratio=1.0, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.neg_edge_ratio = neg_edge_ratio

    @overrides
    def forward(self, data):
        adj_mat = pyg.utils.to_dense_adj(data.edge_index).squeeze()
        out_adj_mat = adj_mat.clone()  # create a masked version of the adjacency matrix
        dim, _ = adj_mat.shape  # get num of nodes

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
        # [ 1 ][ 0 ]
        # ----------
        # [ 0 ][ 0 ]
        # mask quadrant top-right & bottom-right
        out_adj_mat[:, int(dim * self.ratio):] = MASK_VALUE
        # mask quadrant bottom-left
        out_adj_mat[int(dim * self.ratio):, : int(dim * self.ratio)] = MASK_VALUE
        out_edge_index, out_edge_weight = pyg.utils.dense_to_sparse(out_adj_mat)

        # sample negative edges only from the observed (non-masked) region
        # my method is to create a temporary adjacency matrix with the same shape as the original one
        # with the masked region set to 1 and use the negative_sampling function to sample negative edges
        # there may be better way to do this
        temp_adj_mat = adj_mat.clone()
        temp_adj_mat[:, int(dim * self.ratio):] = 1
        temp_adj_mat[int(dim * self.ratio):, : int(dim * self.ratio)] = 1
        pos_edges_observed_out, _ = pyg.utils.dense_to_sparse(temp_adj_mat)
        # the output is pyg.data.Data object but with additional attributes "neg_edge_index"
        out = Data(data.x, edge_index=out_edge_index, edge_weight=out_edge_weight)
        # for input only mask the bottom-right quadrant
        inp_adj_mat = adj_mat.clone()
        # mask quadrant top-left
        # [ 0 ][ 1 ]
        # ----------
        # [ 1 ][ 1 ]
        inp_adj_mat[:int(dim * self.ratio), :int(dim * self.ratio)] = MASK_VALUE
        inp_edge_index, inp_edge_weight = pyg.utils.dense_to_sparse(inp_adj_mat)
        inp = Data(data.x, edge_index=inp_edge_index, edge_weight=inp_edge_weight)
        sample = {'input': inp, 'output': out}
        return sample

# # currently, MaskAdjacencyMatrix is not a functional transform, so it cannot be used in pre_transform since
# # it require return a Data object while MaskAdjacencyMatrix returns a dict of two Data objects.
# # this function is not used in the current implementation
# class pre_transform(BaseTransform):
#     def __init__(self, neg_edge_ratio=1.0):
#         super().__init__()
#         self.to_undirected = ToUndirected()
#         self.mask_adjacency_matrix = MaskAdjacencyMatrix(neg_edge_ratio=neg_edge_ratio)
#         raise NotImplementedError("This function is still under development.")
#     def forward(self, data: Any) -> Any:
#         return self.mask_adjacency_matrix(self.to_undirected(data))


class PermuteNode(BaseTransform):
    def __init__(self, seed=0):
        super().__init__()

    def forward(self, data: Any) -> Any:
        dim = data.x.size(0)
        perm = np.eye(dim, dtype=int)
        np.random.shuffle(perm)
        data.x = data.x[np.where(perm == 1)[1], :]
        # permute the edge_index
        perm = torch.tensor(perm, dtype=torch.float)
        adj_mat = pyg.utils.to_dense_adj(data.edge_index)
        adj_mat = perm @ adj_mat @ perm.T
        adj_mat = adj_mat.squeeze(0)
        edge_index, edge_weight = pyg.utils.dense_to_sparse(adj_mat)
        data.edge_index = edge_index
        return data

#  Random remove edges from the output subgraph
# todo:
class OutputRandomEdgesSplit(BaseTransform):
    def __init__(self, num_val=0.1,  num_test=0.2, neg_sampling_ratio=1.0):
        super().__init__()
        self.random_link_split = RandomLinkSplit(is_undirected=True,  key='edge_label',
                          num_val=num_val, num_test=num_test,
                                                 neg_sampling_ratio=neg_sampling_ratio)



    def forward(self, data: Any) -> Any:
        '''
        :param data:  dictionary of "input" and " output" data returned by
        MaskAdjacencyMatrix
        :return: same dictionary of "input" and "output" data with output data has
        additional edge attributes for indicate its belongs to one of
        train,val, and test sets.
        '''
        output_train, output_val,  output_test  = self.random_link_split(data['output'])
        output = {'train': output_train, 'val': output_val, 'test': output_test}
        return {'input': data['input'], 'output': output}

def get_data(root='.', dataset_name:str = None,
             mask_ratio=0.5,
             num_val=0.1, num_test=0.2,
             neg_edge_ratio=1.0):

    pre_transforms = [ToUndirected()]
    permute_node = PermuteNode()
    pre_transforms.append(permute_node)

    # defne a function that will iterate the input over the list of pre_transforms
    def pre_transform_function(data):
        for transform in pre_transforms:
            data = transform(data)
        return data

    mask_adjacency_matrix = MaskAdjacencyMatrix(ratio=mask_ratio)
    output_random_edge_split = OutputRandomEdgesSplit(num_val=num_val,
                                                      num_test=num_test, neg_sampling_ratio=neg_edge_ratio)
    def transform_function(data):
        return output_random_edge_split(mask_adjacency_matrix(data))

    # load data
    if dataset_name == 'KarateClub':
            dataset = KarateClub(transform=transform_function)
    if dataset_name == 'Cora':
        dataset = Planetoid(root=root, name='Cora', pre_transform=pre_transform_function,
                            transform=transform_function)
    if dataset_name == 'CiteSeer':
        dataset = Planetoid(root=root, name='CiteSeer', pre_transform=pre_transform_function,
                            transform=transform_function)
    if dataset_name == 'PubMed':
        dataset = Planetoid(root=root, name='PubMed', pre_transform=pre_transform_function,
                            transform=transform_function)

    # set shuffle to False to keep the order of the dataset otherwise
    # the split will be different for each epoch
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # dataset_sizes = len(dataset)
    # return dataloader, dataset_sizes

    return dataset[0]