"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/10/24
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from overrides import overrides
from torch_geometric.nn import GCNConv
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import torch_geometric as pyg

from src.utils import ResidualAdd

EPS = 1e-15
MAX_LOGSTD = 10

class BaselineNet(pyg.nn.GAE):
    '''
    The purpose of this class is to provide a baseline model for the CGVAE.
    '''
    def __init__(self, num_node_features, hidden_size, latent_size):
        # use two layers of GCNConv to encode the graph nodes
        encoder = pyg.nn.Sequential('x, edge_index, edge_weight', [
            (GCNConv(in_channels=num_node_features, out_channels=hidden_size), 'x, edge_index, edge_weight -> x1'),
            # (torch.nn.Linear(num_node_features, hidden_size), 'x -> x'),
            # (ResidualAdd(), 'x1, x -> x1'),
            (torch.nn.ReLU(), 'x1 -> x1'),
            # another layer of GCNConv
            (GCNConv(in_channels=num_node_features, out_channels=hidden_size), 'x, edge_index, edge_weight -> x1'),
            (torch.nn.ReLU(), 'x1 -> x1'),
            (GCNConv(in_channels=hidden_size, out_channels=latent_size),'x1, edge_index, edge_weight -> x1')
        ])
        # self.conv1 = GCNConv(in_channels=num_node_features, out_channels=hidden1_size)
        # self.conv2 = GCNConv(in_channels=hidden1_size, out_channels=hidden2_size)
        # use InnerProductDecoder to decode the graph nodes
        decoder = InnerProductDecoder()
        super().__init__(encoder=encoder, decoder=decoder)

    @overrides
    def forward(self, input: pyg.data.Data, *args, **kwargs) -> Tensor:
        z = self.encode(input.x, input.edge_index, input.edge_weight, **kwargs)
        out = self.decode(z, sigmoid=False, **kwargs)
        return out # out is the entire adjacency matrix

    @overrides
    def encode(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, *args, **kwargs) -> Tensor:
        return self.encoder(x, edge_index, edge_weight)

    @overrides
    def decode(self, z: Tensor, sigmoid: bool=False, *args, **kwargs) -> Tensor:
        return self.decoder.forward_all(z, sigmoid) # decode all edges in the graph


# todo:
class MaskedReconstructionLoss(torch.nn.Module):
    def __init__(self, negative_sampling_ratio: float=1.0):
        super().__init__()
        self.negative_sampling_ratio = negative_sampling_ratio

    def forward(self, input: Tensor, target: pyg.data.Data, *args, **kwargs) -> Tensor:
        # target is the torch_geometric.data.Data object
        # input and target['edge_index'] are both adjacency matrices with complimentary masked region
        # the input of baseline model is already an adjacency matrix with continuous values [0, 1]
        # only need to convert the target to an adjacency matrix

        pos_edge_index = target['edge_index']
        num_pos_edge = pos_edge_index.size(1)
        neg_edge_index = negative_sampling(pos_edge_index,
                                           num_nodes=target.num_nodes,
                                           num_neg_samples=self.negative_sampling_ratio * num_pos_edge)
        num_neg_edge = neg_edge_index.size(1)

        loss_edge_index = torch.cat((pos_edge_index, neg_edge_index), dim=-1)
        loss_target = torch.cat((torch.ones(num_pos_edge, dtype=torch.float, device=input.device),
                                 torch.zeros(num_neg_edge, dtype=torch.float, device=input.device)),
                                dim=-1)
        # use BCEWITHLOGITSLOSS to avoid numerical instability
        # Todo: assuming the adjacency matrix of input and target are the same in terms of node order (row and col)
        #   is it safe?
        loss = F.binary_cross_entropy_with_logits(input[loss_edge_index[0], loss_edge_index[1]], loss_target).mean()
        return loss


def train(device, dataloader, num_node_features, learning_rate, num_epochs, model_path):
    '''
    The purpose of this function is to train the baseline model for the CGVAE.
    '''
    # Train baseline
    baseline_net = BaselineNet(num_node_features=num_node_features, hidden_size=32, latent_size=32)
    baseline_net.to(device)
    optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate)
    # negative sampling ratio is important for sparse adjacency matrix
    criterion = MaskedReconstructionLoss(negative_sampling_ratio=1)
    best_loss = np.inf
    early_stop_count = 0

    for epoch in range(num_epochs):
        baseline_net.train()
        running_loss = 0.0
        num_preds = 0

        bar = tqdm(dataloader, desc='NN Epoch {}'.format(epoch).ljust(20))
        for i, batch in enumerate(bar):
            inputs = batch['input'].to(device)
            outputs = batch['output'].to(device)

            optimizer.zero_grad()

            preds = baseline_net(inputs)
            loss = criterion(preds, outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_preds += 1
            if i % 10 == 0:
                bar.set_postfix(loss='{:.2f}'.format(running_loss / num_preds))

        epoch_loss = running_loss # the magnitude of the loss decided by number of edges [node^2]
        print(f'Loss: {epoch_loss:.4f}')

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        # torch.save(baseline_net.state_dict(), model_path)

    return baseline_net