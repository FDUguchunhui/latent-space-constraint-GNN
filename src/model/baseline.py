"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/10/24
"""
import copy
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from torch import Tensor
from overrides import overrides
from torch_geometric.nn import GCNConv, TransformerConv, GIN, GENConv
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import torch_geometric as pyg
import torch.nn.functional as F

#todo: fix baseline model, check whether it predicts the correct adjacency matrix
#todo: baseline perform not good, maybe try use node task learning embedding
# and then use the embedding to predict the edges, this way, the baseline model can
# complete avoid false positive edges in the graph!!!!!
class BaselineNet(pyg.nn.GAE):

    class GCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 2*out_channels)
            # self.conv2 = TransformerConv(2 * out_channels, 2 * out_channels) #todo:  add skip connection
            self.conv3 = GCNConv(2*out_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            # x = self.conv2(x, edge_index).relu()
            x = self.conv3(x, edge_index)
            return x

    '''
    The purpose of this class is to provide a baseline model for the LSCGNN.
    '''
    def __init__(self, num_node_features,out_channels, **kwargs):
        #todo: for baseline maybe try some other architectures than GCN since
        # the predicted adjacency matrix is not clear enough to be used as input

        # use two layers of GCNConv to encode the graph nodes
        # encoder = pyg.nn.Sequential('x, edge_index', [
        #     (GCNConv(in_channels=num_node_features, out_channels=hidden_size), 'x, edge_index -> x1'),
        #     # (torch.nn.Linear(num_node_features, hidden_size), 'x -> x'),
        #     # (ResidualAdd(), 'x1, x -> x1'),
        #     (torch.nn.ReLU(), 'x1 -> x1'),
        #     # another layer of GCNConv
        #     (GCNConv(in_channels=num_node_features, out_channels=hidden_size), 'x, edge_index -> x1'),
        #     (torch.nn.ReLU(), 'x1 -> x1'),
        #     (GCNConv(in_channels=hidden_size, out_channels=latent_size),'x1, edge_index -> x1')
        # ])
        # self.conv1 = GCNConv(in_channels=num_node_features, out_channels=hidden1_size)
        # self.conv2 = GCNConv(in_channels=hidden1_size, out_channels=hidden2_size)
        # use InnerProductDecoder to decode the graph nodes


        encoder = self.GCNEncoder(num_node_features, out_channels)
        # encoder = GIN(num_node_features, hidden_channels=32,
        #               out_channels=16, num_layers=2)

        decoder = InnerProductDecoder()
        super().__init__(encoder=encoder, decoder=decoder)

    @overrides
    def forward(self, input: pyg.data.Data, output_edge_index, *args, **kwargs) -> Tensor:
        z = self.encode(input, output_edge_index, **kwargs)
        out = self.decode(z,  output_edge_index, sigmoid=False, **kwargs)
        return out # out is the entire adjacency matrix

    @overrides
    def encode(self, masked_x: pyg.data.Data, y_edge_index: Tensor, *args, **kwargs) -> Tensor:
        if y_edge_index is not None:
            combined_edge_index = torch.cat((masked_x.edge_index, y_edge_index), dim=1)
        else:
            combined_edge_index = masked_x.edge_index
        return self.encoder(masked_x.x, combined_edge_index)

    @overrides
    def decode(self, z: Tensor, edge_index: Tensor, sigmoid: bool=False, *args, **kwargs) -> Tensor:
        # return self.decoder.forward_all(z, sigmoid) # decode all edges in the graph
        logits = self.decoder(z, edge_index, sigmoid=sigmoid)
        return logits

    def predict(self, input: pyg.data.Data, *args, **kwargs) -> Tensor:
        ''''
        get binary prediction of the edges in the graph for a given subset of nodes
        '''
        z = self.encode(input.x, input.edge_index, input.edge_weight, **kwargs)
        sigmoid_output = self.decoder.forward_all(z, sigmoid=False, **kwargs)
        # make sigmoid_output as 1 if it is greater than 0.5, otherwise 0
        return (sigmoid_output > 0.99).float()

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
        edge_index = torch.cat((pos_edge_index, neg_edge_index), dim=1)
        logits = self.decode(z, edge_index, sigmoid=False)
        # create target labels
        true_labels = torch.zeros(logits.size(0), dtype=torch.float, device=logits.device)
        true_labels[:pos_edge_index.size(1)] = 1
        loss = F.binary_cross_entropy_with_logits(logits, true_labels, reduction='mean')
        return loss


def train(device, data, num_node_features, model_path, learning_rate=10e-3,
          num_epochs=100, early_stop_patience=10, out_channels=16,
          neg_sample_ratio=1, split_ratio=0.5):
    '''
    The purpose of this function is to train the baseline model for the LSCGNN.
    '''
    # Train baseline
    baseline_net = BaselineNet(num_node_features=num_node_features,
                               out_channels=out_channels)
    baseline_net.to(device)
    optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate)
    # negative sampling ratio is important for sparse adjacency matrix
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
                        # some problem with training baseline model
                        z = baseline_net.encode(input, output_train.edge_index)
                        neg_edge_index = negative_sampling(output_train.pos_edge_label_index,
                                                           num_nodes=int(z.size(0) * 0.5),
                                                           num_neg_samples=int(output_train.pos_edge_label_index.size(1) *  neg_sample_ratio) )
                        loss = baseline_net.recon_loss(z, output_train.pos_edge_label_index,
                                                    neg_edge_index= neg_edge_index)
                        loss = loss
                    else:
                        z = baseline_net.encode(input, input.edge_index)
                        loss = baseline_net.recon_loss(z, output_train.pos_edge_label_index,
                                                       neg_edge_index=output_train.neg_edge_label_index)
                        # calculate the predicted logits
                        # predicted_logits = baseline_net.decode(z, output_val.edge_label_index, sigmoid=False)
                        # roc_auc = roc_auc_score(output_val.edge_label, predicted_logits)
                        # calculate average precision
                        # precision, recall, _ = precision_recall_curve(output_val.edge_label, predicted_logits)
                        # average_precision = average_precision_score(output_val.edge_label, predicted_logits)
                        # print the roc_auc and average precision
                        # print(f'ROC AUC: {roc_auc}, Average Precision: {average_precision}')

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

    logging.info(f'Best epoch: {best_epoch}, best loss: {best_loss}')

    # save the final model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_net.state_dict(), model_path)

    return baseline_net

def test( model: BaselineNet, data, device='cpu'):
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
        z = model.encode(input, input.edge_index)
        predicted_logits = model.decoder(z, edge_label_index)  # output should be generate from latent space for test
        # calculate roc_auc
        roc_auc = roc_auc_score(edge_label, predicted_logits)
        # calculate average precision
        precision, recall, _ = precision_recall_curve(edge_label, predicted_logits)
        average_precision = average_precision_score(edge_label, predicted_logits)

    return roc_auc, average_precision
