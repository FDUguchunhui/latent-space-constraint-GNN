"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/11/24
"""
import copy
import logging

import numpy as np
from pathlib import Path
import torch
import torch_geometric as pyg
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from torch import Tensor
from torch_geometric.nn import GCNConv, InnerProductDecoder
from tqdm import tqdm
from src.cgvae.baseline import BaselineNet
from src.cgvae.utils import MaskedReconstructionLoss
import torch.nn.functional as F



MASK_VALUE = 0
EPS = 1e-15
MAX_LOGSTD = 10

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, latent_size):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv_mu = GCNConv(hidden_size, latent_size)
        self.conv_logstd = GCNConv(hidden_size, latent_size)

    def forward(self, masked_x: pyg.data.Data, y_edge_index: Tensor):
        # put x and y together in the same adjacency matrix for simplification
        # x is the input masked graph (with some edges masked and those masked edges are revealed in y),
        # y is the target masked graph (the complement of x)

        # extract edge_index and edge_weight from masked_y only in the observed region
        completed_edge_index = torch.cat((masked_x.edge_index, y_edge_index), dim=1)
        # remove duplicate edges
        # completed_edge_index, _ = pyg.utils.remove_self_loops(completed_edge_index)
        hidden = self.conv1(masked_x.x, completed_edge_index).relu()
        z_mu = self.conv_mu(hidden, completed_edge_index)
        z_logstd = self.conv_logstd(hidden, completed_edge_index)
        return z_mu, z_logstd

class Decoder(torch.nn.Module):
    def __init__(self, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.inner_prod_decoder = InnerProductDecoder()

    def forward(self, z, edge_index: Tensor):
        # return self.inner_prod_decoder.forward_all(z, sigmoid=self.sigmoid)
        return self.inner_prod_decoder(z, edge_index, sigmoid=self.sigmoid)

class CGVAE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_size: int,
                 latent_size: int, pre_treained_baseline_net: BaselineNet,
                 split_ratio=0.5):
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

        # get prior
        if self.predicted_y_edge is None:
            with torch.no_grad():
                #todo: check predict instead of sigmoid output
                edge_label_logits= self.baseline_net(masked_x, masked_y.edge_index) # y_hat is initial predicted adjacency matrix
                y_hat = torch.sigmoid(edge_label_logits)
                # predict 0 or 1
                y_hat = (y_hat > 0.5).bool()
                self.predicted_y_edge = masked_y.edge_label_index[:, y_hat]
                # tuple to tensor
                # predicted_y_edge = torch.stack(predicted_y_edge, dim=0)
                # self.predicted_y_edge =  predicted_y_edge.squeeze()
            #todo: baseline is too noise, maybe try use indirect link directly

        self.prior_mu, self.prior_logstd = self.prior_net(masked_x, self.predicted_y_edge)

        # get posterior
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
        prior_variance = self.prior_logstd.exp()**2
        #
        kl = -0.5 * torch.mean(
            torch.sum(1 + 2 * (self.posterior_logstd - self.prior_logstd)
                      - ((self.posterior_mu - self.prior_mu)**2) * (torch.reciprocal(prior_variance))
                      - (posterior_variance/prior_variance), dim=1))
        return kl

        # return -0.5 * torch.mean(
        #     torch.sum(1 + 2 * self.posterior_logstd - self.posterior_mu**2 - self.posterior_logstd.exp()**2, dim=1))


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

def train(device, data, num_node_features,
          pre_trained_baseline_net,
          model_path,
          out_channels=16,
          learning_rate=10e-3,
          num_epochs=100,
          early_stop_patience=10,
          regularization=1.0):

    cgvae_net = CGVAE(in_channels=num_node_features, hidden_size=2 * out_channels,
                      latent_size=out_channels, pre_treained_baseline_net=pre_trained_baseline_net)
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
                        loss = F.binary_cross_entropy_with_logits(
                            cgvae_net.generation_net(z, output_train.edge_label_index),
                            output_train.edge_label).mean()
                    else:
                        z = cgvae_net(input, output_val)
                        loss = F.binary_cross_entropy_with_logits(
                            cgvae_net.generation_net(z, output_val.edge_label_index),
                                                     output_val.edge_label, reduction='mean')
                    # todo: the KL need to be adjusted by size of output
                    loss = loss + regularization * (1/(input.size(0) * cgvae_net.split_ratio**2)) * cgvae_net.kl_divergence()

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
        # todo: problem here
        output_test = data['output']['test'].to(device)
        z = model(input, output_test)
        predicted_logits = model.generation_net(z, output_test.edge_label_index)
        # predicted_logits = model.generate(output_test.edge_label_index)
        # sample from the conditional posterior and calculate AUC
        # output_edge_label = output_test.edge_label.int().cpu().numpy()
        # pred_logits = model.generation_net(z, output_test.edge_label_index).cpu().numpy()
        # calculate AUC for each batch for output_logits and output_labels
        roc_auc = roc_auc_score(output_test.edge_label, predicted_logits)
        # calculate average precision
        precision, recall, _ = precision_recall_curve(output_test.edge_label, predicted_logits)
        average_precision = average_precision_score(output_test.edge_label, predicted_logits)

    return roc_auc, average_precision
