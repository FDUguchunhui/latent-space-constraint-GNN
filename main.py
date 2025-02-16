import os
import time
import json
import torch
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import os.path as osp
import logging

from src import model
from src.encoder import ReconEncoder, RegEncoder, MLPClassifier
import hydra
from omegaconf import DictConfig

# todo: use hydra for config management
@hydra.main(config_path="./config/homogeneous", config_name="config", version_base='1.3')
def main(cfg: DictConfig):
    # create logger
    logging.basicConfig(level=logging.INFO)

    pyg.seed.seed_everything(cfg.seed)

    # initalize dataloader
    data = model.data_transform.get_data('data/', cfg.data.dataset,
                                         mask_ratio=cfg.data.split_ratio,
                                         num_val=cfg.data.num_val,
                                         num_test=cfg.data.num_test,
                                         neg_sample_ratio=cfg.data.neg_sample_ratio,
                                         false_pos_edge_ratio=cfg.data.false_pos_edge_ratio)

    # count run time from here
    time_start = time.time()

    if cfg.model.layer_type == 'GCNConv':
        conv_layer = GCNConv
    elif cfg.model.layer_type == 'SAGEConv':
        conv_layer = SAGEConv
    else:
        conv_layer = GATConv

    reg_encoder = RegEncoder(conv_layer=conv_layer, hidden_size=cfg.model.out_channels * 2, latent_size=cfg.model.out_channels)
    recon_encoder = ReconEncoder(conv_layer=conv_layer, use_edge_for_predict=cfg.model.use_edge_for_predict, hidden_size=cfg.model.out_channels * 2, latent_size=cfg.model.out_channels)
    num_classes = data.y.max().item() + 1
    classifier = MLPClassifier(input_dim=cfg.model.out_channels, hidden_dim=cfg.model.out_channels * 2, output_dim=num_classes)

    cgvae_net, val_loss = model.cgvae_train(
        device= 'cuda' if torch.cuda.is_available() else 'cpu',
        data=data,
        classifer=classifier,
        reg_encoder=reg_encoder,
        recon_encoder=recon_encoder,
        out_channels=cfg.model.out_channels,
        learning_rate=cfg.train.learning_rate,
        num_epochs=cfg.train.num_epochs,
        model_path=osp.join('checkpoints', str(cfg.seed), 'cgvae_net.pth'),
        regularization=cfg.train.regularization,
        split_ratio=cfg.data.split_ratio,
        neg_sample_ratio=cfg.data.neg_sample_ratio
    )

    end_time = time.time()
    execution_time = end_time - time_start

    accuracy = model.cgvae_model.test(cgvae_net, data)
    print(f'seed: {cfg.seed}, accuracy: {accuracy}')

    # Create a dictionary with the data you want to save
    result_data = {
        'dataset': cfg.data.dataset,
        'split_ratio': cfg.data.split_ratio,
        'use_edge_for_predict': cfg.model.use_edge_for_predict,
        'seed': cfg.seed,
        'val_loss': round(val_loss, 4),
        'accuracy': round(accuracy, 4),
        'learning_rate': cfg.train.learning_rate,
        'regularization': cfg.train.regularization,
        'neg_sample_ratio': round(cfg.data.neg_sample_ratio, 2),
        'false_pos_edge_ratio': round(cfg.data.false_pos_edge_ratio, 2),
        'num_epochs': cfg.train.num_epochs,
        'num_val': cfg.data.num_val,
        'num_test': cfg.data.num_test,
        'execution_time': round(execution_time, 2),
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