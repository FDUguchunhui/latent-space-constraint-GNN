import argparse
import json
import logging
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import torch.optim as optim
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToUndirected
import torch_geometric as pyg
from torch_geometric.nn import VGAE
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
pyg.seed.seed_everything(123)


EPS = 1e-15
# Unconditional generation
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VGAEWithGenerate(VGAE):
    def __init__(self, encoder):
        super().__init__(encoder)
        self.encoder = encoder

    def generate(self):
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return self.decoder.forward_all(z)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))


from pathlib import Path
from tqdm import tqdm
import torch.optim as optim

def train_model(model, epoch, learning_rate, early_stop_patience, model_path, device='cpu'):
    model.to(device)

    early_stop_count = 0
    best_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = np.Inf

    for epoch in range(epoch):
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            with tqdm(total=1, desc='VGAE Epoch {}'.format(epoch).ljust(20)) as bar:
                if phase == 'train':
                    batch = train_data.to(device)
                else:
                    batch = val_data.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    z = model.encode(batch.x, batch.edge_index)
                    # logits = model.decoder(z, batch.pos_edge_label_index)
                    # loss = F.binary_cross_entropy_with_logits(logits, torch.ones(batch.pos_edge_label_index.size(1)), reduction='mean')
                    pos_loss = -torch.log(
                        model.decoder(z, batch.pos_edge_label_index, sigmoid=True) + EPS).mean()
                    neg_edge_index = negative_sampling(batch.pos_edge_label_index, z.size(0))
                    neg_loss = -torch.log(1 -
                                          model.decoder(z, neg_edge_index, sigmoid=True) +
                                          EPS).mean()
                    loss = pos_loss + neg_loss
                    # loss = model.recon_loss(z, train_data.pos_edge_label_index)

                    loss = loss + (1 / batch.size(0)) * model.kl_loss()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                bar.set_postfix(phase=phase, loss='{:.4f}'.format(loss),
                                early_stop_count=early_stop_count)

                if phase == 'val':
                    # save the best loss
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_epoch = epoch
                        torch.save(model.state_dict(), model_path)
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    print(f'Best epoch: {best_epoch}')

    model.load(model_path)
    model.eval()

    # save the final model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def test_model(model, data, device):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return auc, ap


if __name__ == '__main__':

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='CGVAE')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--num_val', type=float, default=0.1)
    parser.add_argument('--num_test', type=float, default=0.2)
    parser.add_argument('--neg_edge_ratio', type=float, default=1)
    # model train arguments
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--out_channels', type=int, default=16)
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--regularization', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=np.Inf)
    # other arguments
    parser.add_argument('--results', type=str, default='results/vgae_results.json')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Set random seed
    pyg.seed.seed_everything(args.seed)

    # Load dataset
    random_link_split = RandomLinkSplit(is_undirected=True,
                                        num_val=0.1, num_test=0.2,
                                        neg_sampling_ratio=1,
                                        split_labels=True,
                                        add_negative_train_samples=False)
    if args.dataset == 'Cora':
        dataset = Planetoid(root='data/vgae_data', name='Cora',
                        transform= T.Compose([
                            T.ToUndirected(),
                            T.NormalizeFeatures(), random_link_split]))
    if args.dataset == 'CiteSeer':
        dataset = Planetoid(root='data/vgae_data', name='CiteSeer',
                        transform= T.Compose([
                            T.ToUndirected(),
                            T.NormalizeFeatures(), random_link_split]))
    if args.dataset == 'PubMed':
        dataset = Planetoid(root='data/vgae_data', name='PubMed',
                        transform= T.Compose([
                            T.ToUndirected(),
                            T.NormalizeFeatures(), random_link_split]))
    if dataset is None:
        raise ValueError('Dataset not found')

    train_data, val_data, test_data = dataset[0]

    vgae_model = VGAEWithGenerate(VariationalGCNEncoder(
        dataset.num_node_features, args.out_channels))
    model = vgae_model.to(args.device)

    start_time = time.time()
    train_model(model, epoch=args.num_epochs,
                learning_rate=args.learning_rate,
                early_stop_patience=args.early_stop_patience,
                model_path='checkpoints/vgae_net.pth',
                device=args.device)
    execution_time = time.time() - start_time

    auc, ap = test_model(model, data=test_data, device=args.device)

    # logging the average AUC and average precision
    print(f'AUC: {auc}, AP: {ap}')


    # Create a dictionary with the data you want to save
    data = {
        'dataset': args.dataset,
        'seed': args.seed,
        'AUC': round(auc, 4),
        'AP': round(ap, 4),
        'execution_time': round(execution_time, 2),
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
    }

    # Read the existing data
    with open(args.results, 'a') as f:
        f.write('\n')
        json.dump(data, f)
