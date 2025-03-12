import argparse
import os
import time
import json
import torch
import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import os.path as osp
import logging
from src.encoder import ReconEncoder, RegEncoder, MLPClassifier
import hydra
from omegaconf import DictConfig

from deeprobust.graph.defense import GCN, ProGNN

from src.model import LSC
from src.data.utils import GraphData


@hydra.main(config_path="./config/homogeneous", config_name="config", version_base='1.3')
def main(cfg: DictConfig):
    # create logger
    logging.basicConfig(level=logging.INFO)

    pyg.seed.seed_everything(cfg.seed)

    # hydra multirun with
    cfg.data.perturb_rate = round(cfg.data.perturb_rate, 2)

    data_generator = GraphData(root='data', save_dir='cache')
    data = data_generator.load_graph(name=cfg.data.dataset,
                                     target_ratio=cfg.data.target_ratio,
                                     perturb_rate=cfg.data.perturb_rate,
                                     perturb_type=cfg.data.perturb_type)
    data = data.to('cuda' if torch.cuda.is_available() else 'cpu')

    # count run time from here
    time_start = time.time()

    pyg.seed.seed_everything(cfg.seed)

    if cfg.model.layer_type == 'GCNConv':
        conv_layer = GCNConv
    elif cfg.model.layer_type == 'SAGEConv':
        conv_layer = SAGEConv
    elif cfg.model.layer_type == 'GATConv':
        conv_layer = GATConv
    else:
        raise ValueError('Invalid layer type')

    reg_encoder = RegEncoder(conv_layer=conv_layer, hidden_size=cfg.model.out_channels * 2, latent_size=cfg.model.out_channels)
    recon_encoder = ReconEncoder(conv_layer=conv_layer, use_edge_for_predict=cfg.model.use_edge_for_predict, hidden_size=cfg.model.out_channels * 2, latent_size=cfg.model.out_channels)
    num_classes = data.y.max().item() + 1
    classifier = MLPClassifier(input_dim=cfg.model.out_channels, hidden_dim=cfg.model.out_channels * 2, output_dim=num_classes)

    if cfg.model.model_type == 'ProGNN':
        args = argparse.Namespace(
            debug=cfg.verbose,
            only_gcn=False,
            lr=0.01,
            weight_decay=5e-4,
            hidden=32,
            ptb_rate=0.05,
            epochs=400,
            alpha=5e-4,
            beta=1.5,
            gamma=1,
            lambda_=0,
            phi=0,
            inner_steps=2,
            outer_steps=1,
            lr_adj=0.01,
            symmetric=False
        )
        model = GCN(nfeat=data.x.size(1),
                    nhid=cfg.model.out_channels,
                    nclass=data.y.max().item() + 1,
                    dropout=False, device='cuda' if torch.cuda.is_available() else 'cpu')

        prognn = ProGNN(model, args, device='cuda' if torch.cuda.is_available() else 'cpu')
        data.edge_index = torch.cat([data.edge_index, data.reg_edge_index], dim=1)
        # get idx from mask where mask is True
        idx_train = data.train_mask.nonzero().view(-1)
        idx_val= data.test_mask.nonzero().view(-1)
        idx_test = data.test_mask.nonzero().view(-1)
        adj = to_dense_adj(data.edge_index)[0]
        prognn.fit(data.x, adj, data.y, idx_train, idx_val)
        val_loss = prognn.best_val_loss.cpu().item()
        accuracy = prognn.test(data.x, data.y, idx_test)

    else:
        cgvae_net, val_loss = LSC.train(
            device= 'cuda' if torch.cuda.is_available() else 'cpu',
            data=data,
            model_type=cfg.model.model_type,
            task_head=classifier,
            reg_encoder=reg_encoder,
            recon_encoder=recon_encoder,
            out_channels=cfg.model.out_channels,
            learning_rate=cfg.train.learning_rate,
            num_epochs=cfg.train.num_epochs,
            model_path=osp.join('checkpoints', str(cfg.seed), 'cgvae_net.pth'),
            regularization=cfg.train.regularization,
            verbose=cfg.verbose  # Add verbose parameter
        )
        accuracy = LSC.test(cgvae_net, data, device= 'cuda' if torch.cuda.is_available() else 'cpu')

    execution_time =  time.time() - time_start


    print(f'seed: {cfg.seed}, accuracy: {accuracy}')

    # Create a dictionary with the data you want to save
    result_data = {
        'dataset': cfg.data.dataset,
        'target_ratio': cfg.data.target_ratio,
        'use_edge_for_predict': cfg.model.use_edge_for_predict,
        'model_type': cfg.model.model_type,
        'layer_type': cfg.model.layer_type,
        'seed': cfg.seed,
        'val_loss': round(val_loss, 4),
        'accuracy': round(accuracy, 4),
        'learning_rate': cfg.train.learning_rate,
        'regularization': cfg.train.regularization,
        'neg_sample_ratio': round(cfg.data.neg_sample_ratio, 2),
        'perturb_type': cfg.data.perturb_type,
        'perturb_rate': round(cfg.data.perturb_rate, 2),
        'num_epochs': cfg.train.num_epochs,
        'execution_time': round(execution_time, 4),
        'time_stamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Ensure the directory part of the results path exists
    os.makedirs(osp.dirname(cfg.results), exist_ok=True)

    # Read the existing data
    with open(cfg.results, 'a') as f:
        f.write('\n')
        json.dump(result_data, f)

if __name__ == '__main__':
    main()