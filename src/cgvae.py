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
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from data_transform import get_data
from src.baseline import BaselineNet

MASK_VALUE = 0

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, latent_size):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv_mu = GCNConv(hidden_size, latent_size)
        self.conv_logstd = GCNConv(hidden_size, latent_size)

    def forward(self, masked_x: pyg.data.Data, masked_y: Tensor):
        # put x and y together in the same adjacency matrix for simplification
        # x is the input masked graph (with some edges masked and those masked edges are revealed in y),
        # y is the target masked graph (the complement of x)

        # extract edge_index and edge_weight from masked_y only in the observed region
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

    def model(self, xs: pyg.data.Data, ys: pyg.data.Data):
        # register PyTorch module `decoder` with Pyro
        pyro.module("generation_net", self)
        with pyro.plate("data"):
            # Prior network uses the baseline predictions as initial guess.
            # This is the generative process with recurrent connection
            with torch.no_grad():
                y_hat = self.baseline_net(xs) # y_hat is initial predicted adjacency matrix

            # simulate the latent variable z from the prior distribution, which is
            # modeled by the input xs (the initial guess y_hat is also a function of xs)

            prior_mu, prior_logstd = self.prior_net(xs, y_hat)
            # for each node in the graph, sample a latent code z from the prior distribution
            zs = torch.empty(xs.num_nodes, self.latent_size)
            for i in range(xs.num_nodes):
                zs[i] = pyro.sample("z_{}".format(i), dist.Normal(prior_mu[i], torch.exp(prior_logstd[i])).to_event(1))

            # the output y is generated from the distribution pθ(y|x, z)
            # generated_adj is generated adjacency matrix with continuous values [0, 1] cells representing
            # the probability of the edge being present
            generated_adj = self.generation_net(zs)

            if ys is not None:
                # In training, we will only sample in the masked graph
                generated_pos_edge_prob = generated_adj[ys.edge_index[0], ys.edge_index[1]]
                # todo: I use negative sampling to sample the negative edges, but I need to make sure
                #  the negative egdes are the same each time I sample
                neg_edge_index = negative_sampling(ys.edge_index, num_nodes=ys.num_nodes,
                                                   num_neg_samples=ys.num_edges)
                generated_neg_edge_prob = generated_adj[neg_edge_index[0], neg_edge_index[1]]
                generated_edge_prob = torch.cat((generated_pos_edge_prob, generated_neg_edge_prob))
                observed_ys = torch.cat((torch.ones(ys.num_edges), torch.zeros(neg_edge_index.size(1))))
                pyro.sample('y', dist.Bernoulli(generated_edge_prob, validate_args=False).to_event(1),
                            obs=observed_ys)
            # else:
            #     # In testing, no need to sample: the output is already a
            #     # probability in [0, 1] range, which better represent pixel
            #     # values considering grayscale. If we sample, we will force
            #     # each pixel to be  either 0 or 1, killing the grayscale
            #     pyro.deterministic('y', loc.detach())

            # return the loc so we can visualize it later
            return generated_adj

    # todo: need to decoder part since it doesn't have learnable parameters
    #   then prior and posterior are the same
    def guide(self, xs: pyg.data.Data, ys: pyg.data.Data = None):
        with pyro.plate("data"):
            if ys is None:
                # at inference time, ys is not provided. In that case,
                # the model uses the prior network
                y_hat = self.baseline_net() # y_hat is initial predicted adjacency matrix
                z_mu, z_logstd = self.prior_net(xs, y_hat)
            else:
                # at training time, uses the variational distribution
                # q(z|x,y) = normal(loc(x,y),scale(x,y))

                # make into adjacency matrix
                ys_adj_mat = pyg.utils.to_dense_adj(ys.edge_index, max_num_nodes=ys.num_nodes, edge_attr=ys.edge_weight)
                z_mu, z_logstd = self.recognition_net(xs, ys_adj_mat)

            zs = torch.empty(xs.num_nodes, self.latent_size)
            for i in range(xs.num_nodes):
                # todo: make sure pyro can handle those type of sample for each node
                zs[i] = pyro.sample("z_{}".format(i), dist.Normal(z_mu[i], torch.exp(z_logstd[i])).to_event(1))



# todo: implement loss calculation
def train(device, dataloader, learning_rate, num_epochs, pre_trained_baseline_net):

    # clear param store
    pyro.clear_param_store()

    # todo: in_channels is hard-coded here need to fix to use difference dataset
    cgvae_net = CGVAE(1433, 50, 50, pre_trained_baseline_net)
    cgvae_net.to(device)
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(cgvae_net.model, cgvae_net.guide, optimizer, loss=Trace_ELBO())

    best_loss = np.inf

    samples = []
    losses = []

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        running_loss = 0.0

        # Iterate over data.
        bar = tqdm(dataloader,
                   desc='CVAE Epoch {}'.format(epoch).ljust(20))
        for i, batch in enumerate(bar):
            inputs = batch['input'].to(device)
            outputs = batch['output'].to(device)

            loss = svi.step(inputs, outputs)

            # statistics
            running_loss += loss
            if i % 10 == 0:
                bar.set_postfix(loss='{:.2f}'.format(loss))

            df = pd.DataFrame(columns=['epoch', 'loss'])
            # df.loc[0] = [epoch + float(i), loss]
            losses.append(df)

        epoch_loss = running_loss

    # # Save model weights
    # cgvae_net.load(model_path)
    #
    # # record evolution
    # samples = pd.concat(samples, axis=0, ignore_index=True)
    # samples.to_csv('samples.csv', index=False)
    #
    # losses = pd.concat(losses, axis=0, ignore_index=True)
    # losses.to_csv('losses.csv', index=False)

    return cgvae_net