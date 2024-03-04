"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/11/24
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn import GCNConv, InnerProductDecoder
from tqdm import tqdm
from data_transform import get_data
from src.baseline import BaselineNet
from utils import MaskedReconstructionLoss

MASK_VALUE = 0
EPS = 1e-15
MAX_LOGSTD = 10

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, latent_size):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv_mu = GCNConv(hidden_size, latent_size)
        self.conv_logstd = GCNConv(hidden_size, latent_size)

    def forward(self, masked_x: pyg.data.Data, masked_y):
        # put x and y together in the same adjacency matrix for simplification
        # x is the input masked graph (with some edges masked and those masked edges are revealed in y),
        # y is the target masked graph (the complement of x)

        # extract edge_index and edge_weight from masked_y only in the observed region
        # todo: for predicted observed region in y, it is not safe to use the edge_index and edge_weight
        #    since it tend to assign non-zero edge_weight to non-existent edges. Need to fix this.
        dim = masked_x.num_nodes
        masked_y[:, dim // 2:] = MASK_VALUE
        # mask quadrant 3
        masked_y[dim // 2:, : dim // 2] = MASK_VALUE
        masked_y_edge_index, masked_y_edge_weight = pyg.utils.dense_to_sparse(masked_y)
        # the edge weight need to be transformed into probability of the edge being present from logit
        masked_y_edge_weight = torch.sigmoid(masked_y_edge_weight)

        completed_edge_index = torch.cat((masked_x.edge_index, masked_y_edge_index), dim=1)
        completed_edge_weight = torch.cat((torch.ones(masked_x.edge_index.size(1)),
                                           masked_y_edge_weight), dim=0)
        # then compute the hidden units
        hidden = self.conv1(masked_x.x, completed_edge_index, completed_edge_weight)
        hidden = torch.relu(hidden)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_mu = self.conv_mu(hidden, completed_edge_index, completed_edge_weight)
        z_logstd = self.conv_logstd(hidden, completed_edge_index, completed_edge_weight)
        return z_mu, z_logstd

class Decoder(torch.nn.Module):
    def __init__(self, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.inner_prod_decoder = InnerProductDecoder()

    def forward(self, z):
        return self.inner_prod_decoder.forward_all(z, sigmoid=self.sigmoid)


class CGVAE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_size: int,
                 latent_size: int, pre_treained_baseline_net: BaselineNet):
        super().__init__()
        # The CGVAE is composed of multiple GNN, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z). Also, CGVAE is built on top of the baselineNet: not only
        # the direct input x, but also the initial guess y_hat made by the baselineNet
        # are fed into the prior network.
        self.latent_size = latent_size
        self.baseline_net = pre_treained_baseline_net
        self.prior_net = Encoder(in_channels, hidden_size, latent_size)
        self.generation_net = Decoder(sigmoid=False)
        self.recognition_net = Encoder(in_channels, hidden_size, latent_size)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, masked_x: pyg.data.Data, masked_y: Tensor):

        # get prior
        with torch.no_grad():
            y_hat = self.baseline_net(masked_x) # y_hat is initial predicted adjacency matrix
        self.prior_mu, self.prior_logstd = self.prior_net(masked_x, y_hat)

        # get posterior
        mask_y_adj_mat = pyg.utils.to_dense_adj(masked_y.edge_index, max_num_nodes=masked_y.num_nodes).squeeze()
        self.posterior_mu, self.posterior_logstd = self.recognition_net(masked_x, mask_y_adj_mat)
        z = self.reparametrize(self.posterior_mu, self.posterior_logstd)
        return z

    def generate(self) -> Tensor:
        with torch.no_grad():
            # sample from the conditional posterior
            z = self.reparametrize(self.posterior_mu, self.posterior_logstd)
            return self.generation_net(z)

    def kl_divergence(self) -> Tensor:
        posterior_variance = self.posterior_logstd.exp()**2
        prior_variance = self.prior_logstd.exp()**2

        kl = -0.5 * torch.mean(
            torch.sum(1 + 2 * (self.posterior_logstd - self.prior_logstd)
                      - (self.posterior_mu - self.prior_mu)**2 * (torch.reciprocal(prior_variance))
                      - (posterior_variance/prior_variance), dim=1))
        return kl

    def save(self, model_path):
        torch.save({'prior': self.prior_net.state_dict(),
                    'generation': self.generation_net.state_dict(),
                    'recognition': self.recognition_net.state_dict()}, model_path)

    def load(self, model_path, map_location=None):
        net_weights = torch.load(model_path, map_location=map_location)
        self.prior_net.load_state_dict(net_weights['prior'])
        self.generation_net.load_state_dict(net_weights['generation'])
        self.recognition_net.load_state_dict(net_weights['recognition'])
        self.prior_net.eval()
        self.generation_net.eval()
        self.recognition_net.eval()

# todo: implement loss calculation
def train(device, dataloader, num_node_features, learning_rate, num_epochs, pre_trained_baseline_net,
          model_path):

    # todo: in_channels is hard-coded here need to fix to use difference dataset
    cgvae_net = CGVAE(in_channels=num_node_features, hidden_size=50,
                      latent_size=50, pre_treained_baseline_net=pre_trained_baseline_net)
    cgvae_net.to(device)
    optimizer = torch.optim.Adam(lr=learning_rate, params=cgvae_net.parameters())
    reconstruction_loss = MaskedReconstructionLoss()
    for epoch in range(num_epochs):
        bar = tqdm(dataloader,
                   desc='CVAE Epoch {}'.format(epoch).ljust(20))
        for i, batch in enumerate(bar):
            inputs = batch['input'].to(device)
            outputs = batch['output'].to(device)

            optimizer.zero_grad()
            cgvae_net.train()

            z = cgvae_net(inputs, outputs)
            recon_loss = reconstruction_loss(cgvae_net.generation_net(z), outputs)
            kl_loss = cgvae_net.kl_divergence()
            loss = recon_loss + (1 / inputs.num_nodes) * kl_loss
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss='{:.4f}'.format(loss))

    # save the model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    cgvae_net.save(model_path)

    return cgvae_net