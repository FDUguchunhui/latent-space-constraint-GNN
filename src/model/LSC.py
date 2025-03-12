"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/11/24
"""

from pathlib import Path
import torch
from torch import Tensor
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import torch.nn.functional as F
from src.gcn_proprocess import GCN_2layer, dropedge_jaccard, truncatedSVD


class LSCGNN(torch.nn.Module):
    def __init__(self, reg_encoder, recon_encoder, task_head, latent_size: int):
        super().__init__()
        # The LSCGNN is composed of multiple GNN, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z). Also, LSCGNN is built on top of the baselineNet: not only
        # the direct input x, but also the initial guess y_hat made by the baselineNet
        # are fed into the prior network.
        self.latent_size = latent_size
        self.reg_net = reg_encoder
        self.recon_net = recon_encoder
        self.task_head =  task_head
        self.predicted_y_edge = None
        # target_ratio is useful for the baselineNet to predict the target part of the adjacency matrix
        # and standardize KL loss for the size of the output

    def forward(self, data):

        # combine masked_y.edge_index and predicted_y_edge to get the complete edge_index
        reg_latent = self.reg_net(data.x,  data.reg_edge_index) # todo: change back
        target_latent = self.recon_net(data)
        return target_latent, reg_latent


def reg_loss(target_latent, reg_latent) -> Tensor:
    kl =  torch.mean(
        torch.mean(((target_latent - reg_latent)**2)
                   , dim=1))

    # if variational
    # kl = -0.5 * torch.mean(
    #     torch.sum(1 + 2 * (self.posterior_logstd[:split_index])
    #               - ((self.posterior_mu[:split_index] - self.prior_mu[:split_index])**2)
    #               - (posterior_variance[:split_index]), dim=1))

    return kl



def train(device,
          data,
          model_path,
          reg_encoder,
          recon_encoder,
          task_head,
          model_type,
          out_channels=16,
          learning_rate=10e-3,
          num_epochs=100,
          regularization=1.0,
          verbose=True):  # Add verbose parameter

    # if model_type == 'LSCGNN':
    model = LSCGNN(reg_encoder=reg_encoder,
                   recon_encoder=recon_encoder,
                   task_head=task_head,
                   latent_size=out_channels)
    if model_type == 'GCNJaccard':
        # model = GCN_2layer(hidden_channels=64, out_channels=32)
        all_edges = torch.cat([data.edge_index, data.reg_edge_index], dim=1)
        data.edge_index = dropedge_jaccard(all_edges, data.x, threshold=0.01)
    elif model_type == 'GCNSVD':
        # model = GCN_2layer(hidden_channels=64, out_channels=32)
        all_edges = torch.cat([data.edge_index, data.reg_edge_index], dim=1)
        all_edges = to_dense_adj(all_edges)[0]
        all_edges = truncatedSVD(all_edges, k=50)
        data.edge_index = dense_to_sparse(torch.tensor(all_edges, device=device))[0]
    else:
        raise ValueError('Invalid model type')

    model.to(device)
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    data = data.to(device)

    for epoch in range(num_epochs):
        train_loss, val_loss = None, None

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            with tqdm(total=1, desc=f'CGVAE Epoch {epoch}', disable=not verbose) as bar:  # Use verbose parameter
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        zs = model(data)
                        if isinstance(zs, tuple):
                            z, reg_z = zs
                        else:
                            z = zs
                            reg_z = None

                        logits = model.task_head(z)
                        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                        if reg_z is not None:
                            loss = loss + regularization * reg_loss(z[data.target_node_mask], reg_z[data.target_node_mask])

                        loss.backward()
                        optimizer.step()
                        train_loss = loss.item()

                    else:
                        zs = model(data)
                        if isinstance(zs, tuple):
                            z, reg_z = zs
                        else:
                            z = zs
                            reg_z = None

                        logits = model.task_head(z)
                        loss = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask])
                        val_loss = loss.item()

                bar.set_postfix(phase=phase, loss='{:.4f}'.format(loss))

    model.eval()
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    return model, val_loss
def test(model: LSCGNN, data, device='cpu'):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        zs = model(data)
        if isinstance(zs, tuple):
            z, reg_z = zs
        else:
            z = zs
            reg_z = None
        predicted_logits = model.task_head(z)
        # multi-class accuracy
        _, predicted_labels = torch.max(predicted_logits, 1)
        true_labels = data.y[data.test_mask]
        accuracy = (predicted_labels[data.test_mask] == true_labels).sum().item() / len(true_labels)
        # AUC
    return accuracy
