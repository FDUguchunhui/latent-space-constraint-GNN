"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/11/24
"""
import copy
import logging
from typing import Optional

import numpy as np
from pathlib import Path
import torch
import torch_geometric as pyg
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from torch import Tensor
from torch_geometric.nn import GCNConv, InnerProductDecoder, GINConv
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from src.cgvae.utils import MaskedReconstructionLoss
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, latent_size):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv_mu = GCNConv(hidden_size, latent_size)
        self.conv_logstd = GCNConv(hidden_size, latent_size)

    def forward(self, masked_x: pyg.data.Data, y_edge_index: Tensor=None):
        # put x and y together in the same adjacency matrix for simplification
        # x is the input masked graph (with some edges masked and those masked edges are revealed in y),
        # y is the target masked graph (the complement of x)

        # for prior network, the input need to be input edge + output edge
        if y_edge_index is not None:
            combined_edge_index = torch.cat((masked_x.edge_index, y_edge_index), dim=1)
        else:
            combined_edge_index = masked_x.edge_index
        # combined_edge_index =  y_edge_index
        # extract edge_index and edge_weight from masked_y only in the observed region
        x = self.conv1(masked_x.x, combined_edge_index).relu()
        z_mu = self.conv_mu(x, combined_edge_index)
        z_logstd = self.conv_logstd(x, combined_edge_index)
        return z_mu, z_logstd

class Encoder2(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, latent_size):
        super().__init__()
        # self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv_mu = GCNConv(in_channels, latent_size)

    def forward(self, masked_x: pyg.data.Data, y_edge_index: Tensor=None):
        # put x and y together in the same adjacency matrix for simplification
        # x is the input masked graph (with some edges masked and those masked edges are revealed in y),
        # y is the target masked graph (the complement of x)

        # for prior network, the input need to be input edge + output edge
        if y_edge_index is not None:
            combined_edge_index = torch.cat((masked_x.edge_index, y_edge_index), dim=1)
        else:
            combined_edge_index = masked_x.edge_index
        # combined_edge_index =  y_edge_index
        # extract edge_index and edge_weight from masked_y only in the observed region
        # x = self.conv1(masked_x.x, combined_edge_index).relu()
        z_mu = self.conv_mu(masked_x.x, combined_edge_index)
        return z_mu


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_prod_decoder = InnerProductDecoder()

    def forward(self, z, edge_index: Tensor, sigmoid=False):
        # return self.inner_prod_decoder.forward_all(z, sigmoid=self.sigmoid)
        return self.inner_prod_decoder(z, edge_index, sigmoid=sigmoid)

class CGVAE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_size: int,
                 latent_size: int,
                 split_ratio=0.5):
        super().__init__()
        # The CGVAE is composed of multiple GNN, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z). Also, CGVAE is built on top of the baselineNet: not only
        # the direct input x, but also the initial guess y_hat made by the baselineNet
        # are fed into the prior network.
        self.latent_size = latent_size
        self.prior_net = Encoder2(in_channels, hidden_size, latent_size)
        self.generation_net = Decoder()
        self.recognition_net = Encoder(in_channels, hidden_size, latent_size)
        self.predicted_y_edge = None
        # split_ratio is useful for the baselineNet to predict the target part of the adjacency matrix
        # and standardize KL loss for the size of the output
        self.split_ratio = split_ratio
    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, masked_x: pyg.data.Data, masked_y: pyg.data.Data):

        # combine masked_y.edge_index and predicted_y_edge to get the complete edge_index
        self.prior_mu = self.prior_net(masked_x) # todo: change back

        # get posterior
        # masked_y_edge_index = torch.cat((masked_y.edge_index, self.predicted_y_edge), dim=1)
        self.posterior_mu, self.posterior_logstd = self.recognition_net(masked_x, masked_y.edge_index)
        z = self.reparametrize(self.posterior_mu, self.posterior_logstd)
        return z

    def generate(self, edge_index) -> Tensor:
        with torch.no_grad():
            # sample from the conditional posterior
            z = self.reparametrize(self.posterior_mu, self.posterior_logstd)
            return self.generation_net(z, edge_index)

    def kl_divergence(self) -> Tensor:
        posterior_variance = self.posterior_logstd.exp()**2
        split_index = int(self.prior_mu.size(0) * self.split_ratio)
        #
        kl = -0.5 * torch.mean(
            torch.sum(1 +
                      - ((self.posterior_mu[:split_index] - self.prior_mu[:split_index])**2)
                      , dim=1))
        return kl

        # return -0.5 * torch.mean(
        #     torch.sum(1 + 2 * self.posterior_logstd - self.posterior_mu**2 - self.posterior_logstd.exp()**2, dim=1))

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """

        # concatenate positive and negative edges
        edge_index = torch.cat((pos_edge_index, neg_edge_index), dim=1)
        logits = self.generation_net(z, edge_index, sigmoid=False)
        # create target labels
        true_labels = torch.zeros(logits.size(0), dtype=torch.float, device=logits.device)
        true_labels[:pos_edge_index.size(1)] = 1
        loss = F.binary_cross_entropy_with_logits(logits, true_labels, reduction='mean')
        return loss

    def save(self, model_path):
        torch.save({'prior': self.prior_net.state_dict(),
                    'generation': self.generation_net.state_dict(),
                    'recognition': self.recognition_net.state_dict(),
                    'posterior_mu': self.posterior_mu,
                    'posterior_logstd': self.posterior_logstd,
                    },
                   model_path)

    def load(self, model_path, map_location=None):
        net_weights = torch.load(model_path, map_location=map_location)
        self.prior_net.load_state_dict(net_weights['prior'])
        self.generation_net.load_state_dict(net_weights['generation'])
        self.recognition_net.load_state_dict(net_weights['recognition'])
        self.posterior_mu = net_weights['posterior_mu']
        self.posterior_logstd = net_weights['posterior_logstd']
        self.prior_net.eval()
        self.generation_net.eval()
        self.recognition_net.eval()

def train(device,
          num_node_features,
          data,
          model_path,
          out_channels=16,
          learning_rate=10e-3,
          num_epochs=100,
          early_stop_patience=10,
          regularization=1.0,
          split_ratio=0.5, neg_sample_ratio=1):

    cgvae_net = CGVAE(in_channels=num_node_features, hidden_size=2 * out_channels,
                      latent_size=out_channels,
                      split_ratio=split_ratio)
    cgvae_net.to(device)
    # only optimize the parameters of the CGVAE
    optimizer = torch.optim.Adam(lr=learning_rate, params=cgvae_net.parameters('recognition_net'))
    best_loss = np.inf
    early_stop_count = 0
    best_epoch = 0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    input = data['input'].to(device)
    output_train = data['output']['train'].to(device)
    output_val = data['output']['val'].to(device)

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                cgvae_net.train()
            else:
                cgvae_net.eval()

            with tqdm(total=1, desc=f'CGVAE Epoch {epoch}') as bar:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        z = cgvae_net(input, output_train)
                        neg_edge_index = negative_sampling(output_train.pos_edge_label_index,
                                                           num_nodes=int(z.size(0) * cgvae_net.split_ratio),
                                                           num_neg_samples=int(output_train.pos_edge_label_index.size(1) *  neg_sample_ratio) )
                        loss = cgvae_net.recon_loss(z, output_train.pos_edge_label_index,
                                                    neg_edge_index=neg_edge_index)
                        loss = loss + regularization * (1/input.size(0)) * cgvae_net.kl_divergence()

                    else:
                        z = cgvae_net(input, output_val)
                        loss = cgvae_net.recon_loss(z, output_val.pos_edge_label_index,
                                                    output_val.neg_edge_label_index)
                    # todo: the KL need to be adjusted by size of output
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
                        best_model_wts = copy.deepcopy(cgvae_net.state_dict())
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    cgvae_net.load_state_dict(best_model_wts)
    cgvae_net.eval()

    # save the final model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cgvae_net.state_dict(), model_path)

    logging.info(f'Best epoch: {best_epoch}')

    return cgvae_net, best_epoch #todo: return tuple may not be good should try logger later

def test( model: CGVAE, data, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        input = data['input'].to(device)
        output_test = data['output']['test'].to(device)
        # todo: problem here
        # create edge_index by combining pos_edge_label_index and neg_edge_label_index
        edge_label_index = torch.cat((output_test.neg_edge_label_index,output_test.pos_edge_label_index), dim=1)
        # create edge_label by combining pos_edge_label and neg_edge_label
        edge_label = torch.ones(
            edge_label_index.size(1), dtype=torch.float, device=edge_label_index.device)
        edge_label[:output_test.neg_edge_label_index.size(1)] = 0
        predicted_logits = model.generate(edge_label_index)  # output should be generate from latent space for test
        # calculate roc_auc
        roc_auc = roc_auc_score(edge_label, predicted_logits)
        # calculate average precision
        precision, recall, _ = precision_recall_curve(edge_label, predicted_logits)
        average_precision = average_precision_score(edge_label, predicted_logits)

    return roc_auc, average_precision
