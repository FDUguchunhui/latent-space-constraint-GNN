import pickle
from typing import Union

import torch
import torch_geometric.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
from torch_geometric.utils import negative_sampling


class HeteroDataModule(pl.LightningDataModule):
    def __init__(self, data_path, target_node_type, target_edge_type, undirected=True, num_val=0.1, num_test=0.2, test_edge_list=None,
                 neg_sample_ratio=1, target_only_in_recon=False, add_noisy_edges: int=0, batch_size=512):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.target_node_type = target_node_type
        self.target_edge_type = target_edge_type
        self.undirected = undirected
        self.num_val = num_val
        self.num_test = num_test
        self.test_edge_list = test_edge_list
        self.neg_sample_ratio = neg_sample_ratio
        self.target_only_in_recon=target_only_in_recon
        self.add_noisy_edges = add_noisy_edges

    def prepare_data(self):
        # Load dataset using pickle
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def setup(self, stage=None):

        if self.add_noisy_edges != 0:
            # sample negative edges and add to the data
            false_neg_edge_index = negative_sampling(self.data[self.target_edge_type]['edge_index'],
                                                     num_nodes=self.data[self.target_node_type].x.size(0),
                                                     num_neg_samples=int(self.data[self.target_edge_type]['edge_index'].size(1) * self.add_noisy_edges))
            self.data[self.target_edge_type]['edge_index'] =  torch.cat([self.data[self.target_edge_type]['edge_index'], false_neg_edge_index], dim=1)

        # Split the data
        if self.undirected:
            undirected_transformer = T.ToUndirected()
            self.data = undirected_transformer(self.data)

        transformer = T.RandomLinkSplit(
            num_val=self.num_val,
            num_test=self.num_test,
            is_undirected=self.undirected,
            neg_sampling_ratio=self.neg_sample_ratio,
            add_negative_train_samples=False,
            edge_types=self.target_edge_type
        )

        # Set reg_graph and full_graph_metadata
        reg_graph = self.data.clone()
        reg_graph[self.target_edge_type]
        self.reg_graph_metadata = reg_graph.metadata()

        # this is for simulation of the case where we only keep target edge type and target node type
        node_types, edge_types =self.data.metadata()

        self.full_graph_metadata = self.data.metadata()

        if self.target_only_in_recon:
            # only keep target edge type and target node type
            for key in node_types:
                if key != self.target_node_type:
                    self.full_graph_metadata[0].remove(key)
            for key in edge_types:
                if key != self.target_edge_type:
                    self.full_graph_metadata[1].remove(key)

        train_data, val_data, test_data = transformer(self.data)
        if self.num_test == 0:
            test_data[self.target_edge_type]['pos_edge_label_index'] = self.test_edge_list
            # random sample negative edges
            neg_edge_label_index = negative_sampling(test_data[self.target_edge_type]['pos_edge_label_index'],
                             num_nodes=test_data['gene'].x.size(0),
                             num_neg_samples=int(self.test_edge_list.size(1) * self.neg_sample_ratio))
            test_data[self.target_edge_type]['neg_edge_label_index'] = neg_edge_label_index

        self.data = {'train': train_data, 'val': val_data, 'test': test_data}

    def train_dataloader(self):
        return LinkNeighborLoader(
            self.data['train'],
            num_neighbors=[10, 10],  # Example: 10 neighbors for 2 layers
            batch_size=self.batch_size,
            edge_label_index=(self.target_edge_type, self.data['train'][self.target_edge_type]['edge_label_index']),
            neg_sampling_ratio=1.0,
            shuffle=True,
        )

    def val_dataloader(self):
        return LinkNeighborLoader(
            self.data['val'],
            num_neighbors=[10, 10],
            batch_size=self.batch_size * 3,
            edge_label_index=(self.target_edge_type, self.data['val'][self.target_edge_type]['edge_label_index']),
            edge_label=self.data['val'][self.target_edge_type]['edge_label'],
            shuffle=False,
        )

    def test_dataloader(self):
        return LinkNeighborLoader(
            self.data['test'],
            num_neighbors=[10, 10],
            batch_size=self.batch_size * 3,
            edge_label_index=(self.target_edge_type, self.data['test'][self.target_edge_type]['edge_label_index']),
            edge_label=self.data['test'][self.target_edge_type]['edge_label'],
            shuffle=False,
        )


class FullDataLoader(DataLoader):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.batch_size = 1
        super().__init__(dataset=[], *args, **kwargs)

    def __iter__(self):
        yield self.data

    def __len__(self):
        return 1
