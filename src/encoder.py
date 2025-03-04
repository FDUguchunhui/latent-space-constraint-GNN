import torch
import torch_geometric as pyg
from torch import Tensor, nn
from torch_geometric.nn import Linear
from torch_geometric.nn.conv import MessagePassing


class ReconEncoder(torch.nn.Module):
    def __init__(self, conv_layer, hidden_size, latent_size, use_edge_for_predict='combined'):
        super().__init__()
        self.use_edge_for_predict = use_edge_for_predict
        self.conv1 = conv_layer(-1, hidden_size)
        self.lin1 = Linear(-1, hidden_size)
        self.conv2 = conv_layer(-1, latent_size)

    def forward(self, data):
        if self.use_edge_for_predict == 'target':
            edge_index = data.edge_index
        elif self.use_edge_for_predict == 'regularization':
            edge_index = data.reg_edge_index
        elif self.use_edge_for_predict == 'full':
            edge_index = torch.cat((data.edge_index, data.reg_edge_index), dim=1)
        # combined_edge_index = masked_x.edge_index
        z = (self.conv1(data.x, edge_index) + self.lin1(data.x)).relu()
        z = self.conv2(z, edge_index)
        return z


class RegEncoder(torch.nn.Module):
    def __init__(self, conv_layer, hidden_size, latent_size):
        super().__init__()
        self.conv1 = conv_layer(-1, hidden_size)
        self.lin1 = Linear(-1, hidden_size)
        self.conv_mu = conv_layer(-1, latent_size)

    def forward(self, x, reg_edge_index):
        # z = (self.conv1(masked_x.x, masked_x.edge_index) + self.lin1(masked_x.x)).relu()
        z = self.conv_mu(x, reg_edge_index)
        return z

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.mlp(z)