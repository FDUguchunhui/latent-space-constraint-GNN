import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = np.Inf

    for epoch in range(epoch):
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            bar = tqdm(dataloader, desc='VGAE Epoch {}'.format(epoch).ljust(20))
            for train, val, test in bar:
                if phase == 'train':
                    batch = train.to(device)
                else:
                    batch = val.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # why zero_grad need to be after set_grad_enabled
                    z = model.encode(batch.x, batch.pos_edge_label_index)
                    loss = model.recon_loss(z, batch.pos_edge_label_index, batch.neg_edge_label_index)
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
                        torch.save(model.state_dict(), model_path)
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    model.load(model_path)
    model.eval()

    # save the final model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if True:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


from sklearn.metrics import roc_auc_score, average_precision_score
pyg.seed.seed_everything(123)

def test_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for train, val, test in dataloader:
            batch = test.to(device)
            logits = model.generate()
            pos_pred_logits = logits[batch.pos_edge_label_index[0], batch.pos_edge_label_index[1]].cpu().numpy()
            neg_pred_logits = logits[batch.neg_edge_label_index[0], batch.neg_edge_label_index[1]].cpu().numpy()
            pred_logits = np.concatenate((pos_pred_logits, neg_pred_logits))
            # create a torch tensor with 0 and 1 for edge and non-edge with
            # length the same as pos_edge_label and neg_edge_label
            true_labels = torch.cat([torch.ones(batch.pos_edge_label.size(0)),
                                     torch.zeros(batch.neg_edge_label.size(0))])
            # calculate AUC for each batch for output_logits and output_labels
            roc_auc = roc_auc_score(true_labels, pred_logits)
            # calculate average precision
            average_precision = average_precision_score(true_labels, pred_logits)

    # print the average AUC and average precision
    print(f"AUC: {roc_auc}, AP: {average_precision}")

    return roc_auc, average_precision


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
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--regularization', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=50)
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
                                        split_labels=True)
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

    # make shuffle=False to keep the order of the dataset, same as CGVAE
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

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

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    auc, ap = test_model(model, dataloader=dataloader, device=args.device)

    # Create a dictionary with the data you want to save
    data = {
        'dataset': args.dataset,
        'seed': args.seed,
        'AUC': round(auc, 4),
        'AP': round(ap, 4),
        'execution_time': round(execution_time, 2)
    }

    # Read the existing data
    with open(args.results, 'a') as f:
        f.write('\n')
        json.dump(data, f)
