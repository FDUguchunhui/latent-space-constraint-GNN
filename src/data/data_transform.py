"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/10/24
"""
import copy
import os
from typing import Any

import scipy.sparse as sp
import torch_geometric as pyg
from overrides import overrides
from torch_geometric.datasets import  AttributedGraphDataset, Yelp, Planetoid, Reddit2
from torch_geometric.transforms import BaseTransform, ToUndirected, AddSelfLoops, RandomLinkSplit, RemoveDuplicatedEdges
import torch_geometric.transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.utils import to_torch_coo_tensor
from src.model.random_graph import RandomGraph
from deeprobust.graph.defense import GCN
from src.data.mettack import MetaApprox, Metattack
from deeprobust.graph.utils import *
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import warnings


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




class RandomEdgePerturbation(BaseTransform):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, data: Data) -> Data:
        if not hasattr(data, 'target_node_mask'):
            raise ValueError("target_node_mask is not found in data. Ensure MaskAdjacencyMatrix is applied first.")

        # Get the indices of the nodes with target_node_mask
        target_node_indices = torch.where(data.target_node_mask)[0]
        num_target_nodes = len(target_node_indices)
        num_total_nodes = data.x.size(0)

        # Convert edge_index to a set of tuples for faster lookup
        existing_edges = set(map(tuple, data.edge_index.t().tolist()))

        # Generate false positive and negative edges
        num_edges = len(existing_edges)
        num_perturb_edges = int(num_edges * self.ratio)
        new_edges = []
        neg_edges = []

        while len(new_edges) + len(neg_edges) < num_perturb_edges:
            i = torch.randint(0, num_target_nodes, (1,)).item()
            j = torch.randint(0, num_target_nodes, (1,)).item()
            edge = (target_node_indices[i].item(), target_node_indices[j].item())
            if edge not in existing_edges and (edge[1], edge[0]) not in existing_edges:
                new_edges.append(edge)
            if edge in existing_edges or (edge[1], edge[0]) in existing_edges:
                neg_edges.append(edge)

        # remove edges in neg_edges from the existing edges
        for edge in neg_edges:
            existing_edges.remove(edge)

        # Add the new edges to the existing edges
        existing_edges.update(new_edges)
        data.edge_index = torch.tensor(list(existing_edges)).t().contiguous()

        return data

class Mettack(BaseTransform):

    def __init__(self, ratio=0.1, device='cuda' if torch.cuda.is_available() else 'cpu', model='Train'):
        super().__init__()
        self.ratio = ratio
        self.device = device
        self.model = model

    def forward(self, data: Data) -> Data:
        if not hasattr(data, 'target_node_mask'):
            raise ValueError("target_node_mask is not found in data. Ensure MaskAdjacencyMatrix is applied first.")

        data = data.to(self.device)

        # Get the indices of the nodes with target_node_mask
        target_node_indices = torch.where(data.target_node_mask)[0]

        adj, features, labels =  to_scipy_sparse_matrix(data.edge_index), data.x, data.y
        idx_train, idx_val, idx_test = torch.nonzero(data.train_mask).squeeze(), torch.nonzero(data.val_mask).squeeze(), torch.nonzero(data.test_mask).squeeze()
        idx_unlabeled = np.union1d(idx_val, idx_test)

        perturbations = int(self.ratio * (adj.sum()))

        # Setup Surrogate Model
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device='cuda' if torch.cuda.is_available() else 'cpu')

        surrogate.fit(features, adj, labels, idx_train)

        # Setup Attack Model
        if 'Self' in self.model:
            lambda_ = 0
        if 'Train' in self.model:
            lambda_ = 1
        if 'Both' in self.model:
            lambda_ = 0.5

        if 'A' in self.model:
            model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                               attack_structure=True, attack_features=False, device=self.device, lambda_=lambda_,
                               train_iters=100)

        else:
            model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                              attack_structure=True, attack_features=False, device=self.device, lambda_=lambda_,
                              train_iters=100)

        model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
        modified_adj = sp.csr_matrix(model.modified_adj)
        data.edge_index = from_scipy_sparse_matrix(modified_adj)

        return data

class RemoveNodeFeature(BaseTransform):
    def __init__(self, ratio=0.5):
        super().__init__()

    def forward(self, data: pyg.data.Data) -> Any:
        # one-hot encoding of the node position with 0 to num_nodes-1
        data.x = torch.eye(data.x.size(0), dtype=torch.float)
        return data

class OutputRandomNodesSplit(BaseTransform):
    def __init__(self, num_val=0.1, num_test=0.2):
        super().__init__()
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
             target_ratio=0.5,
             num_val=0.1, num_test=0.2,
             perturb_rate=0.1,
             perturb_type='Random'
             ):
    '''
    This function returns the dataset object with the specified transformation.
    :param root: the root directory to store the dataset
    :param dataset_name: the name of the dataset
    :param target_ratio: the ratio of the adjacency matrix to be masked
    :param num_val: the ratio of the validation set
    :param num_test: the ratio of the test set
    :param neg_sample_ratio: the ratio of the negative samples
    :param featureless: whether to remove the node features
    :param perturb_rate: whether to add false positive edges, None means not to add
    '''

    pre_transform_functions = [T.NormalizeFeatures()]
    if dataset_name == 'PPI' or dataset_name == 'Facebook':
        transform_y = TransformY()
        pre_transform_functions.append(transform_y)

    # defne a function that will iterate the input over the list of pre_transforms
    pre_transforms = T.Compose(pre_transform_functions)

    mask_adjacency_matrix = MaskAdjacencyMatrix(ratio=target_ratio)
    output_random_node_split = OutputRandomNodesSplit(num_val=num_val, num_test=num_test)

    transform_functions = []

    transform_functions.append(mask_adjacency_matrix)

    transform_functions.append(output_random_node_split)

    if perturb_rate is not None and perturb_rate > 0:
        if perturb_type == 'Mettack':
            transform_functions.append(Mettack(ratio=perturb_rate, device='cuda' if torch.cuda.is_available() else 'cpu'))
        elif perturb_type == 'Random':
            transform_functions.append(RandomEdgePerturbation(ratio=perturb_rate))
        else:
            raise ValueError('Invalid perturbation type')

    transform_functions.append(ToUndirected())
    transforms = T.Compose(transform_functions)

    if dataset_name == 'RandomGraph':
        dataset = RandomGraph(num_nodes=2000, p=0.1, root=root, pre_transform=pre_transforms,
                              transform=transforms)
    elif dataset_name == 'Cora':
        dataset = Planetoid(root=root, name='Cora', pre_transform=pre_transforms,
                            transform=transforms)
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root=root, name='CiteSeer', pre_transform=pre_transforms,
                            transform=transforms)
    elif dataset_name == 'PubMed':
        dataset = Planetoid(root=root, name='PubMed', pre_transform=pre_transforms,
                            transform=transforms)
    #todo: PPI dataset need to be double check, cannot add false-positive edges
    elif dataset_name == 'PPI':
        dataset = AttributedGraphDataset(root=os.path.join(root, 'PPI'), name='PPI',
                                         pre_transform=pre_transforms,
                     transform=transforms)
    elif dataset_name == 'Reddit':
        dataset = Reddit2(root=os.path.join(root, 'Reddit'), pre_transform=pre_transforms,
                     transform=transforms)
    elif dataset_name == 'Facebook':
        dataset = AttributedGraphDataset(root=root, name='Facebook',
                                         pre_transform=pre_transforms,
                     transform=transforms)
    else:
        raise ValueError('Dataset not found')

    # set shuffle to False to keep the order of the dataset otherwise
    # the split will be different for each epoch
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # dataset_sizes = len(dataset)
    # return dataloader, dataset_sizes

    return dataset[0]