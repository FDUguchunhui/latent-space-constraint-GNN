from typing import List

import torch
from networkx import watts_strogatz_graph
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp

from torch_geometric.utils import erdos_renyi_graph, from_networkx


class RandomGraph(InMemoryDataset):
    # generate a random graph based on the Erdos-Renyi model
    def __init__(self, num_nodes, p, root,  type='ws', pre_transform=None,  transform=None):
        self.num_nodes = num_nodes
        self.p = p
        self.root = root
        self.type = type
        super().__init__('.', transform=transform, pre_transform=pre_transform)
        self.load(self.processed_paths[0])
        data = self.get(0)
        self.data, self.slices = self.collate([data])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'RG_{self.num_nodes}_{self.p}', 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self) -> None:
        if self.type == 'ws':
            G = watts_strogatz_graph(self.num_nodes, 4, self.p)
            # convert networkx graph to PyG data
            data = from_networkx(G)
            data.x = torch.eye(self.num_nodes)

        if self.type == 'er':
            edge_index = erdos_renyi_graph(self.num_nodes, self.p)
            data = Data(edge_index=edge_index, x=torch.eye(self.num_nodes))

        self.save([data], self.processed_paths[0])
