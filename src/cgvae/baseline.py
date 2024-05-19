"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/10/24
"""
import copy
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from overrides import overrides
from torch_geometric.nn import GCNConv
from torch_geometric.nn import InnerProductDecoder
from tqdm import tqdm
import torch_geometric as pyg

from src.cgvae.utils import MaskedReconstructionLoss

EPS = 1e-15
MAX_LOGSTD = 10

class BaselineNet(pyg.nn.GAE):
    '''
    The purpose of this class is to provide a baseline model for the CGVAE.
    '''
    def __init__(self, num_node_features, hidden_size, latent_size):
        #todo: for baseline maybe try some other architectures than GCN since
        # the predicted adjacency matrix is not clear enough to be used as input

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
    def forward(self, input: pyg.data.Data, output_edge_index, *args, **kwargs) -> Tensor:
        z = self.encode(input.x, input.edge_index, input.edge_weight, **kwargs)
        out = self.decode(z,  output_edge_index, sigmoid=False, **kwargs)
        return out # out is the entire adjacency matrix

    @overrides
    def encode(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, *args, **kwargs) -> Tensor:
        return self.encoder(x, edge_index, edge_weight)

    @overrides
    def decode(self, z: Tensor, edge_index: Tensor, sigmoid: bool=False, *args, **kwargs) -> Tensor:
        # return self.decoder.forward_all(z, sigmoid) # decode all edges in the graph
        logits = self.decoder(z, edge_index, sigmoid=sigmoid)
        sigmoid_output = torch.sigmoid(logits)
        prediction = (sigmoid_output > 0.5).bool()
        return logits, edge_index[:, prediction] # return the predicted edges (with values > 0.5)

    def predict(self, input: pyg.data.Data, *args, **kwargs) -> Tensor:
        z = self.encode(input.x, input.edge_index, input.edge_weight, **kwargs)
        sigmoid_output = self.decoder.forward_all(z, sigmoid=False, **kwargs)
        # make sigmoid_output as 1 if it is greater than 0.5, otherwise 0
        return (sigmoid_output > 0.5).float()



def train(device, data, num_node_features, model_path, learning_rate=10e-3,
          num_epochs=100, early_stop_patience=10, hidden_size=32, latent_size=16):
    '''
    The purpose of this function is to train the baseline model for the CGVAE.
    '''
    # Train baseline
    baseline_net = BaselineNet(num_node_features=num_node_features,
                               hidden_size=hidden_size, latent_size=latent_size)
    baseline_net.to(device)
    optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate)
    # negative sampling ratio is important for sparse adjacency matrix
    criterion = MaskedReconstructionLoss()
    best_loss = np.inf
    best_epoch = 0
    early_stop_count = 0
    best_model_wts = copy.deepcopy(baseline_net.state_dict())

    # get the input and output data and move them to the device
    input = data['input'].to(device)
    output_train = data['output']['train'].to(device)
    output_val = data['output']['val'].to(device)

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                baseline_net.train()
            else:
                baseline_net.eval()

            with tqdm(total=1, desc=f'NN Epoch {epoch}') as bar:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # only use input to predict the output edges with train mask
                    if phase == 'train':
                        edge_label_logits = baseline_net(input, output_train)
                        loss = criterion(edge_label_logits, output_train.edge_label)
                    else:
                        edge_label_logits = baseline_net(input, output_val)
                        loss = criterion(edge_label_logits, output_val.edge_label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss = loss  # the magnitude of the loss decided by number of edges [node^2]
                bar.set_postfix(phase=phase, loss='{:.4f}'.format(epoch_loss),
                                early_stop_count=early_stop_count)

                if phase == 'val':
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_epoch = epoch
                        best_model_wts = copy.deepcopy(baseline_net.state_dict())
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    baseline_net.load_state_dict(best_model_wts)
    baseline_net.eval()

    # save the final model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_net.state_dict(), model_path)

    return baseline_net