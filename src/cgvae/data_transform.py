"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/10/24
"""
from typing import Any

import torch_geometric as pyg
import torch_geometric.seed
from overrides import overrides
from torch import Tensor
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub, AttributedGraphDataset, Yelp, AmazonProducts, Flickr, CoraFull
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.transforms import BaseTransform, ToUndirected, AddSelfLoops, RandomLinkSplit, RemoveDuplicatedEdges
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import torch_geometric.transforms as T

from src.cgvae.random_graph import RandomGraph


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
        split_index = int(dim * self.ratio)
        # TODO: the rule about what region of adjacency matrix is masked is different from the mask in image since
        #   the adjacency matrix is symmetric and the image is not necessarily symmetric. Current implementation only
        #   masks the bottom-right quadrant of the adjacency matrix. But it should support more masking options later.
        #   nor 1 (existent)
        # [ 1 ][ 0 ]
        # ----------
        # [ 0 ][ 0 ]
        # mask quadrant top-right & bottom-right
        out_adj_mat[:, int(split_index):] = MASK_VALUE
        # mask quadrant bottom-left
        out_adj_mat[int(split_index):, : int(split_index)] = MASK_VALUE
        out_edge_index, out_edge_weight = pyg.utils.dense_to_sparse(out_adj_mat)

        temp_adj_mat = adj_mat.clone()
        temp_adj_mat[:, int(split_index):] = 0
        temp_adj_mat[int(split_index):, : int(split_index)] = 0
        out_edge_index, out_edge_weight = pyg.utils.dense_to_sparse(temp_adj_mat)
        # the output is pyg.data.Data object but with additional attributes "neg_edge_index"
        out = Data(data.x, edge_index=out_edge_index, edge_weight=out_edge_weight)
        # for input only mask the bottom-right quadrant
        inp_adj_mat = adj_mat.clone()
        # mask quadrant top-left
        # [ 0 ][ 1 ]
        # ----------
        # [ 1 ][ 1 ]
        inp_adj_mat[:int(split_index), :int(split_index)] = MASK_VALUE
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


class AddFalsePositiveEdge(BaseTransform):
    #todo: need to keep added false positive fixed for different runs
    def __init__(self, ratio=0.5, false_pos_ratio=1.0):
        super().__init__()
        self.ratio = ratio
        self.false_pos_ratio = false_pos_ratio

    def forward(self, data: dict) -> Any:
        output = data['output']
        split_index = int(output.num_nodes * self.ratio)
        false_pos_edge_index = pyg.utils.negative_sampling(data['output'].edge_index,
                                                           num_nodes=split_index,
                                                           num_neg_samples=int(output.edge_index.size(1) * self.false_pos_ratio))
        output.edge_index = torch.cat([output.edge_index, false_pos_edge_index], dim=1)
        return {'input': data['input'], 'output': output, 'false_pos_edge_index': false_pos_edge_index}

class RemoveNodeFeature(BaseTransform):
    def __init__(self, ratio=0.5):
        super().__init__()

    def forward(self, data: pyg.data.Data) -> Any:
        # one-hot encoding of the node position with 0 to num_nodes-1
        data.x = torch.eye(data.x.size(0), dtype=torch.float)
        return data


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
    def __init__(self, num_val=0.1,  num_test=0.2, neg_sampling_ratio=1.0,
                 add_input_edges_to_output=False):
        super().__init__()
        self.random_link_split = RandomLinkSplit(is_undirected=True,  key='edge_label',
                                                 num_val=num_val, num_test=num_test,
                                                 neg_sampling_ratio=neg_sampling_ratio,
                                                 split_labels=True,
                                                 add_negative_train_samples=True)
        self.add_input_edges_to_output = add_input_edges_to_output

    def forward(self, data: Any) -> Any:
        '''
        :param data:  dictionary of "input" and " output" data returned by
        MaskAdjacencyMatrix
        :return: same dictionary of "input" and "output" data with output data has
        additional edge attributes for indicate its belongs to one of
        train,val, and test sets.
        '''
        output_train, output_val, output_test = self.random_link_split(data['output'])

        # check is data has key call false_pos_edge
        if 'false_pos_edge' in data:
            num_nodes = data['input'].x.size(0)

            # remove element in edge1 that is in edge2
            edge1 = pyg.utils.to_dense_adj(output_test['pos_edge_label_index'],
                                           max_num_nodes=num_nodes).squeeze()
            edge2 = pyg.utils.to_dense_adj(data['false_pos_edge_index'],
                                           max_num_nodes=num_nodes).squeeze()
            mask = edge1 - edge2
            mask[mask != 1] = 0

            output_test['pos_edge_label_index'], _ = pyg.utils.dense_to_sparse(mask)
            # sample remove same length of negative edges from the test data
            output_test['neg_edge_label_index'] = output_test['neg_edge_label_index'][:output_test['pos_edge_label_index'].size(1)]

        if self.add_input_edges_to_output:
            output_train['edge_index'] = torch.concat([output_train['edge_index'], data['input'].edge_index], dim=1)
            output_val['edge_index'] = torch.concat([output_train['edge_index'], data['input'].edge_index], dim=1)
            output_test['edge_index'] = torch.concat([output_train['edge_index'], data['input'].edge_index], dim=1)

        output = {'train': output_train, 'val': output_val, 'test': output_test}

        return {'input': data['input'], 'output': output}

def get_data(root='.', dataset_name:str = None,
             mask_ratio=0.5,
             num_val=0.1, num_test=0.2,
             neg_sample_ratio=1.0,
             featureless=False,
             false_pos_edge_ratio=1.0,
             add_input_edges_to_output=False
             ):
    '''
    This function returns the dataset object with the specified transformation.
    :param root: the root directory to store the dataset
    :param dataset_name: the name of the dataset
    :param mask_ratio: the ratio of the adjacency matrix to be masked
    :param num_val: the ratio of the validation set
    :param num_test: the ratio of the test set
    :param neg_sample_ratio: the ratio of the negative samples
    :param featureless: whether to remove the node features
    :param false_pos_edge_ratio: whether to add false positive edges, None means not to add
    '''

    pre_transform_functions = [T.NormalizeFeatures(), ToUndirected()]

    # defne a function that will iterate the input over the list of pre_transforms
    pre_transforms = T.Compose(pre_transform_functions)

    mask_adjacency_matrix = MaskAdjacencyMatrix(ratio=mask_ratio)
    output_random_edge_split = OutputRandomEdgesSplit(num_val=num_val,
                                                      num_test=num_test,
                                                      neg_sampling_ratio=neg_sample_ratio,
                                                      add_input_edges_to_output=add_input_edges_to_output)
    permute_node = PermuteNode()

    transform_functions = []

    if featureless:
        transform_functions.append(RemoveNodeFeature())

    transform_functions = transform_functions + [
        # PermuteNode(),
        mask_adjacency_matrix,
    ]

    if false_pos_edge_ratio is not None and false_pos_edge_ratio > 0:
        transform_functions.append(AddFalsePositiveEdge(ratio=mask_ratio, false_pos_ratio=false_pos_edge_ratio))

    transform_functions.append(output_random_edge_split)
    transforms = T.Compose(transform_functions)

    if dataset_name == 'RandomGraph':
        dataset = RandomGraph(num_nodes=2000, p=0.1, root=root, pre_transform=pre_transforms,
                              transform=transforms)
    if dataset_name == 'Cora':
        dataset = Planetoid(root=root, name='Cora', pre_transform=pre_transforms,
                            transform=transforms)
    if dataset_name == 'CiteSeer':
        dataset = Planetoid(root=root, name='CiteSeer', pre_transform=pre_transforms,
                            transform=transforms)
    if dataset_name == 'PubMed':
        dataset = Planetoid(root=root, name='PubMed', pre_transform=pre_transforms,
                            transform=transforms)
    if dataset_name == 'PPI':
        dataset = AttributedGraphDataset(root=root, name='PPI',
                                         pre_transform=pre_transforms,
                      transform=transforms)
    if dataset_name == 'facebook':
        dataset = AttributedGraphDataset(root=root, name='facebook',
                                         pre_transform=pre_transforms,
                      transform=transforms)
    if dataset_name == 'Yelp':
        dataset = CoraFull(root=root, pre_transform=pre_transforms,
                      transform=transforms)
    if dataset is None:
        raise ValueError('Dataset not found')

    # set shuffle to False to keep the order of the dataset otherwise
    # the split will be different for each epoch
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # dataset_sizes = len(dataset)
    # return dataloader, dataset_sizes

    return dataset[0]