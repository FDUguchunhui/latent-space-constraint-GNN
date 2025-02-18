"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/10/24
"""
import copy
from typing import Any

import torch_geometric as pyg
import torch_geometric.seed
from overrides import overrides
from torch import Tensor
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub, AttributedGraphDataset, Yelp, AmazonProducts, Flickr, CoraFull, Amazon
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.transforms import BaseTransform, ToUndirected, AddSelfLoops, RandomLinkSplit, RemoveDuplicatedEdges
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import torch_geometric.transforms as T

from src.model.random_graph import RandomGraph


# @functional_transform('mask_adjacency_matrix')
class MaskAdjacencyMatrix(BaseTransform):
    '''
    This transformation masks the adjacency matrix of torch_geometric dataset.
    '''

    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio

    @overrides
    def forward(self, data):
        num_nodes = data.x.size(0)
        # randomly draw self.ratio of the nodes id start from 0 to num_nodes-1
        reg_node_ids = np.random.choice(num_nodes, int(num_nodes * (1-self.ratio)), replace=False)
        # create a boolean mask to indicate the nodes to be masked
        data.target_node_mask = torch.tensor(~np.isin(np.arange(num_nodes), reg_node_ids), dtype=torch.bool)

        # remove edge_index related reg_node_id from edge_index and create a reg_edge_index
        # Convert edge_index to numpy array for easier manipulation
        edge_index_np = data.edge_index.numpy()
        # Create a boolean mask to filter out edges with source or target in node_list
        mask = ~np.isin(edge_index_np[0], reg_node_ids) & ~np.isin(edge_index_np[1], reg_node_ids)
        # Apply the mask to edge_index
        data.edge_index = torch.tensor(edge_index_np[:, mask], dtype=torch.long)
        data.reg_edge_index = torch.tensor(edge_index_np[:, ~mask], dtype=torch.long)

        return data


class AddFalsePositiveEdge(BaseTransform):
    #todo: need to keep added false positive fixed for different runs
    def __init__(self, ratio=0.5, false_pos_ratio=1.0):
        super().__init__()
        self.ratio = ratio
        self.false_pos_ratio = false_pos_ratio

    def forward(self, data: Data) -> Any:
        split_index = int(data.num_nodes * self.ratio)
        false_pos_edge_index = pyg.utils.negative_sampling(data.edge_index,
                                                           num_nodes=split_index,
                                                           num_neg_samples=int(data.edge_index.size(1) * self.false_pos_ratio))
        data.edge_index = torch.cat([data.edge_index, false_pos_edge_index], dim=1)
        return data

class RemoveNodeFeature(BaseTransform):
    def __init__(self, ratio=0.5):
        super().__init__()

    def forward(self, data: pyg.data.Data) -> Any:
        # one-hot encoding of the node position with 0 to num_nodes-1
        data.x = torch.eye(data.x.size(0), dtype=torch.float)
        return data

class OutputRandomNodesSplit(BaseTransform):
    def __init__(self, split_ratio, num_val=0.1, num_test=0.2):
        super().__init__()
        self.split_ratio = split_ratio
        self.num_val = num_val
        self.num_test = num_test


    def forward(self, data: Any) -> Any:
        num_nodes = data.x.size(0)
        num_target_nodes = int(data.target_node_mask.sum())
        num_val = int(num_target_nodes * self.num_val)
        num_test = int(num_target_nodes * self.num_test)
        num_train = num_target_nodes - num_val - num_test

        # Get the indices of the unmasked nodes
        unmasked_indices = torch.where(data.target_node_mask)[0]

        # Permute only the unmasked nodes
        perm = unmasked_indices[torch.randperm(num_target_nodes)]

        train_idx = perm[:num_train]
        val_idx = perm[num_train:num_train + num_val]
        test_idx = perm[num_train + num_val:]

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True

        return data


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

        if self.add_input_edges_to_output:
            output_train['edge_index'] = torch.concat([output_train['edge_index'], data['input'].edge_index], dim=1)
            output_val['edge_index'] = torch.concat([output_train['edge_index'], data['input'].edge_index], dim=1)
            output_test['edge_index'] = torch.concat([output_train['edge_index'], data['input'].edge_index], dim=1)

        output = {'train': output_train, 'val': output_val, 'test': output_test}

        return {'input': data['input'], 'output': output}


class TransformY(BaseTransform):
    def forward(self, data: dict) -> Any:
        # transform the y to int instead of one-hot encoding
        data.y = torch.argmax(data.y, dim=1)
        return data

def get_data(root='.', dataset_name:str = None,
             mask_ratio=0.5,
             num_val=0.1, num_test=0.2,
             neg_sample_ratio=1.0,
             false_pos_edge_ratio=1.0,
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
    mask_adjacency_matrix = MaskAdjacencyMatrix(ratio=mask_ratio)
    output_random_node_split = OutputRandomNodesSplit(num_val=num_val, num_test=num_test, split_ratio=mask_ratio)

    pre_transform_functions = [T.NormalizeFeatures(), ToUndirected(), mask_adjacency_matrix, output_random_node_split]

    if false_pos_edge_ratio is not None and false_pos_edge_ratio > 0:
        pre_transform_functions.append(AddFalsePositiveEdge(ratio=mask_ratio, false_pos_ratio=false_pos_edge_ratio))

    if dataset_name == 'PPI':
            transform_y = TransformY()
            pre_transform_functions.append(transform_y)

    # defne a function that will iterate the input over the list of pre_transforms
    pre_transforms = T.Compose(pre_transform_functions)

    if dataset_name == 'RandomGraph':
        dataset = RandomGraph(num_nodes=2000, p=0.1, root=root, pre_transform=pre_transforms)
    if dataset_name == 'Cora':
        dataset = Planetoid(root=root, name='Cora', pre_transform=pre_transforms)
    if dataset_name == 'CiteSeer':
        dataset = Planetoid(root=root, name='CiteSeer', pre_transform=pre_transforms)
    if dataset_name == 'PubMed':
        dataset = Planetoid(root=root, name='PubMed', pre_transform=pre_transforms)

    if dataset is None:
        raise ValueError('Dataset not found')

    # set shuffle to False to keep the order of the dataset otherwise
    # the split will be different for each epoch
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # dataset_sizes = len(dataset)
    # return dataloader, dataset_sizes

    return dataset[0]