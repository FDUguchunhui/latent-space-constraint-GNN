import argparse

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

def train_model(model, epoch, learning_rate, early_stop_patience, model_path, device):
    early_stop_count = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = np.Inf

    for epoch in range(epoch):
        for phase in ['train', 'val']:
            bar = tqdm(dataloader, desc='VGAE Epoch {}'.format(epoch).ljust(20))
            for train, val, test in bar:
                if phase == 'train':
                    batch = train.to(device)
                    model.train()
                else:
                    batch = val.to(device)
                    model.eval()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
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


def test_model(model, dataloader, device):
    all_auc = []
    model.eval()
    with torch.no_grad():
        for train, val, test in dataloader:
            batch = test.to(device)
            pred_logits_list = []

            for i in range(100):
                logits = model.generate()
                pos_pred_logits = logits[batch.pos_edge_label_index[0], batch.pos_edge_label_index[1]].cpu().numpy()
                neg_pred_logits = logits[batch.neg_edge_label_index[0], batch.neg_edge_label_index[1]].cpu().numpy()
                pred_logits_list.append(np.concatenate((pos_pred_logits, neg_pred_logits)))
                # take average of predicted logits
            mean_pred_logits = np.mean(pred_logits_list, axis=0)
            # create a torch tensor with 0 and 1 for edge and non-edge with
            # length the same as pos_edge_label and neg_edge_label
            true_labels = torch.cat([torch.ones(batch.pos_edge_label.size(0)),
                                     torch.zeros(batch.neg_edge_label.size(0))])
            # calculate AUC for each batch for output_logits and output_labels
            roc_auc = roc_auc_score(true_labels,  mean_pred_logits)
            average_precision = average_precision_score(true_labels, mean_pred_logits)
    all_auc.append(roc_auc)

    print(f"Mean AUC: {np.mean(all_auc)}")

test_model(model)

if __name__ == "__main__":

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='CGVAE')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--num_val', type=float, default=0.1)
    parser.add_argument('--num_test', type=float, default=0.2)
    parser.add_argument('--neg_edge_ratio', type=float, default=1)
    # model train arguments
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=64)
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--regularization', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    # other arguments
    parser.add_argument('--results', type=str, default='results/results.json')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Set random seed
    pyg.seed.seed_everything(args.seed)

    # Load dataset
    random_link_split = RandomLinkSplit(is_undirected=True,
                                        num_val=args.num_val,
                                        num_test=args.num_test,
                                        neg_sampling_ratio=args.neg_edge_ratio,
                                        split_labels=True)
    dataset = Planetoid(root='data/cora2', name=args.dataset, pre_transform=ToUndirected(),
                        transform= random_link_split)
    # make shuffle=False to keep the order of the dataset, same as CGVAE
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    vgae_model = VGAEWithGenerate(VariationalGCNEncoder(dataset.num_node_features, dataset.num_node_features*2))
    model = vgae_model.to(args.device)

    train_model(model, 300, 0.001,
                50,
                'checkpoints/vgae_net.pth',
                device=args.device)

