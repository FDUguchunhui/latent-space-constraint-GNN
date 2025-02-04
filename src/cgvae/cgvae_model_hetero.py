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
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from torch import Tensor
from torch_geometric.nn import InnerProductDecoder, to_hetero, SAGEConv
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl

class recon_encoder(torch.nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_size)
        self.conv2 = SAGEConv((-1, -1), latent_size)

    def forward(self, x, edge_index):
        # put x and y together in the same adjacency matrix for simplification
        # x is the input masked graph (with some edges masked and those masked edges are revealed in y),
        # y is the target masked graph (the complement of x)

        # for prior network, the input need to be input edge + output edge
        # combined_edge_index =  y_edge_index
        # extract edge_index and edge_weight from masked_y only in the observed region
        z = self.conv1(x, edge_index).relu()
        z = self.conv2(z, edge_index)
        return z

class reg_encoder(torch.nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_size)
        # self.conv2 = SAGEConv((-1, -1), latent_size)

    def forward(self, x, edge_index):
        # todo: investigate when using only one layer the performance is better
        # try add a skip connection
        z = self.conv1(x, edge_index)
        # z = self.conv2(z, edge_index)
        return z


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_prod_decoder = InnerProductDecoder()

    def forward(self, z, edge_index: Tensor, sigmoid=False):
        # return self.inner_prod_decoder.forward_all(z, sigmoid=self.sigmoid)
        return self.inner_prod_decoder(z, edge_index, sigmoid=sigmoid)

class HeteroCGVAELightning(pl.LightningModule):
    def __init__(self, hidden_size: int,
                 latent_size: int, reg_graph, full_graph_metadata, neg_sample_ratio,
                 target_node_type, regularization, target_edge_type, learning_rate, seed):
        super().__init__()
        # The CGVAE is composed of multiple GNN, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z). Also, CGVAE is built on top of the baselineNet: not only
        # the direct input x, but also the initial guess y_hat made by the baselineNet
        # are fed into the prior network.
        self.save_hyperparameters(ignore=['full_graph_metadata', 'reg_graph'])
        self.reg_graph = reg_graph
        self.reg_net = reg_encoder(hidden_size, latent_size)
        self.reg_net = to_hetero(self.reg_net, self.reg_graph.metadata(), aggr='sum')
        self.generation_net = Decoder()
        self.recon_net = recon_encoder(hidden_size, latent_size)
        self.recon_net = to_hetero(self.recon_net, full_graph_metadata, aggr='sum')
        self.best_val_loss = float('inf')

    def forward(self, full_graph) -> Tensor:

        self.prior = self.reg_net(self.reg_graph.x_dict, self.reg_graph.edge_index_dict)[self.hparams.target_node_type]
        self.posterior = self.recon_net(full_graph.x_dict, full_graph.edge_index_dict)[self.hparams.target_node_type]
        return self.posterior

    def on_fit_start(self) -> None:
       pl.seed_everything(self.hparams.seed)
       self.logger.log_hyperparams(self.hparams, {"val_loss": 0, "best_val_loss": 0, "test_roc_auc": 0})

    def training_step(self, batch, batch_idx):
        z = self(batch)
        neg_edge_label_index = negative_sampling(batch[self.hparams.target_edge_type].pos_edge_label_index,
                                           num_nodes=batch[self.hparams.target_node_type].x.size(0),
                                           num_neg_samples=int(batch[self.hparams.target_edge_type].pos_edge_label_index.size(1) * self.hparams.neg_sample_ratio))
        loss = self.recon_loss(z, batch[self.hparams.target_edge_type].pos_edge_label_index,
                                neg_edge_index=neg_edge_label_index)
        loss = loss + self.hparams.regularization * (1/batch[self.hparams.target_node_type].x.size(0)) * self.reg_loss()
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)


        return loss

    def validation_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.recon_loss(z, batch[self.hparams.target_edge_type].pos_edge_label_index, batch[self.hparams.target_edge_type].neg_edge_label_index)
        edge_label_index = torch.cat((batch[self.hparams.target_edge_type].neg_edge_label_index,
                                      batch[self.hparams.target_edge_type].pos_edge_label_index), dim=1)
        # create edge_label by combining pos_edge_label and neg_edge_label
        edge_label = torch.ones(edge_label_index.size(1), dtype=torch.float, device=edge_label_index.device)
        edge_label[:batch[self.hparams.target_edge_type].neg_edge_label_index.size(1)] = 0
        predicted_logits = self.generate(edge_label_index)  # output should be generated from latent space for test
        # calculate roc_auc
        roc_auc = roc_auc_score(edge_label, predicted_logits)
        values = {'val_loss': loss, 'val_roc_auc': roc_auc}
        if loss is not None and loss < self.best_val_loss:
            self.best_val_loss = loss
        self.log_dict(values, on_epoch=True, batch_size=1, prog_bar=True)
        return loss, roc_auc

    def test_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.recon_loss(z, batch[self.hparams.target_edge_type].pos_edge_label_index, batch[self.hparams.target_edge_type].neg_edge_label_index)
        # calculate roc_auc
        edge_label_index = torch.cat((batch[self.hparams.target_edge_type].neg_edge_label_index,
                                      batch[self.hparams.target_edge_type].pos_edge_label_index), dim=1)
        # create edge_label by combining pos_edge_label and neg_edge_label
        edge_label = torch.ones(edge_label_index.size(1), dtype=torch.float, device=edge_label_index.device)
        edge_label[:batch[self.hparams.target_edge_type].neg_edge_label_index.size(1)] = 0
        predicted_logits = self.generate(edge_label_index)  # output should be generated from latent space for test
        # calculate roc_auc
        roc_auc = roc_auc_score(edge_label, predicted_logits)
        values = {'test_loss': loss, 'test_roc_auc': roc_auc, 'best_val_loss': self.best_val_loss}
        self.log_dict(values, batch_size=1,  prog_bar=True)
        return loss, roc_auc


    def configure_optimizers(self):
        return torch.optim.Adam(lr=self.hparams.learning_rate, params=self.parameters())


    def generate(self, edge_index) -> Tensor:
        with torch.no_grad():
            # sample from the conditional posterior
            return self.generation_net(self.posterior, edge_index)

    def reg_loss(self) -> Tensor:
        loss = torch.mean(torch.sum(((self.posterior - self.prior)**2), dim=1))
        return loss

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

#     def save(self, model_path):
#         torch.save({'prior': self.reg_net.state_dict(),
#                     'generation': self.generation_net.state_dict(),
#                     'recognition': self.recon_net.state_dict(),
#                     'posterior': self.posterior,
#                     },
#                    model_path)
#
#     def load(self, model_path, map_location=None):
#         net_weights = torch.load(model_path, map_location=map_location)
#         self.reg_net.load_state_dict(net_weights['prior'])
#         self.generation_net.load_state_dict(net_weights['generation'])
#         self.recon_net.load_state_dict(net_weights['recognition'])
#         self.posterior = net_weights['posterior']
#         self.reg_net.eval()
#         self.generation_net.eval()
#         self.recon_net.eval()
# #
# def train(device,
#           num_node_features,
#           data,
#           target_node_type,
#           target_edge_type,
#           model_path,
#           out_channels=16,
#           learning_rate=10e-3,
#           num_epochs=100,
#           early_stop_patience=10,
#           regularization=1.0,
#           neg_sample_ratio=1):
#
#     cgvae_net = HeteroCGVAE(in_channels=num_node_features, hidden_size=2 * out_channels,
#                       latent_size=out_channels, data=data, target_node_type=target_node_type, target_edge_type=target_edge_type)
#     # only optimize the parameters of the CGVAE
#     optimizer = torch.optim.Adam(lr=learning_rate, params=cgvae_net.parameters('recognition_net'))
#     best_loss = np.inf
#     early_stop_count = 0
#     best_epoch = 0
#     Path(model_path).parent.mkdir(parents=True, exist_ok=True)
#
#     train_data = data['train'].to(device)
#     # create a regularization graph by removing the target edge type
#     val_data = data['val'].to(device)
#     cgvae_net = cgvae_net.to(device)
#
#     for epoch in range(num_epochs):
#
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 cgvae_net.train()
#             else:
#                 cgvae_net.eval()
#
#             with tqdm(total=1, desc=f'CGVAE Epoch {epoch}') as bar:
#                 optimizer.zero_grad()
#                 with torch.set_grad_enabled(phase == 'train'):
#                     if phase == 'train':
#                         z = cgvae_net(train_data)
#                         neg_edge_label_index = negative_sampling(train_data[target_edge_type].pos_edge_label_index,
#                                                            num_nodes=train_data[target_node_type].x.size(0),
#                                                            num_neg_samples=int(train_data[target_edge_type].pos_edge_label_index.size(1) *  neg_sample_ratio) )
#                         loss = cgvae_net.recon_loss(z, train_data[target_edge_type].pos_edge_label_index,
#                                                     neg_edge_index=neg_edge_label_index)
#                         loss = loss + regularization * (1/train_data[target_node_type].x.size(0)) * cgvae_net.kl_divergence()
#
#                     else:
#                         z = cgvae_net(val_data)
#                         loss = cgvae_net.recon_loss(z, val_data[target_edge_type].pos_edge_label_index,
#                                                     val_data[target_edge_type].neg_edge_label_index)
#                     # todo: the KL need to be adjusted by size of output
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 epoch_loss = loss  # the magnitude of the loss decided by number of edges [node^2]
#                 bar.set_postfix(phase=phase, loss='{:.4f}'.format(epoch_loss),
#                                 early_stop_count=early_stop_count)
#
#                 if phase == 'val':
#                     if epoch_loss < best_loss:
#                         best_loss = epoch_loss
#                         best_epoch = epoch
#                         best_model_wts = copy.deepcopy(cgvae_net.state_dict())
#                         early_stop_count = 0
#                     else:
#                         early_stop_count += 1
#
#         if early_stop_count >= early_stop_patience:
#             break
#
#     cgvae_net.load_state_dict(best_model_wts)
#     cgvae_net.eval()
#
#     # save the final model
#     Path(model_path).parent.mkdir(parents=True, exist_ok=True)
#     torch.save(cgvae_net.state_dict(), model_path)
#
#     logging.info(f'Best epoch: {best_epoch}, best loss: {best_loss:.4f}')
#
#     return cgvae_net, best_epoch, best_loss #todo: return tuple may not be good should try logger later
#
# def test(model, data, target_node_type, target_edge_type, device='cpu'):
#
#     test_data = data['test'].to(device)
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         # create edge_index by combining pos_edge_label_index and neg_edge_label_index
#         edge_label_index = torch.cat((test_data[target_edge_type].neg_edge_label_index,
#                                       test_data[target_edge_type].pos_edge_label_index), dim=1)
#         # create edge_label by combining pos_edge_label and neg_edge_label
#         edge_label = torch.ones(
#             edge_label_index.size(1), dtype=torch.float, device=edge_label_index.device)
#         edge_label[:test_data[target_edge_type].neg_edge_label_index.size(1)] = 0
#         predicted_logits = model.generate(edge_label_index)  # output should be generated from latent space for test
#         # calculate roc_auc
#         roc_auc = roc_auc_score(edge_label, predicted_logits)
#         # calculate average precision
#         precision, recall, _ = precision_recall_curve(edge_label, predicted_logits)
#         average_precision = average_precision_score(edge_label, predicted_logits)
#
#     return roc_auc, average_precision
