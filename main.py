import time
import json
import torch
import torch_geometric as pyg
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

import  src.cgvae as cgvae
import argparse
import os.path as osp
import logging

from src.cgvae.encoder.encoder import ReconEncoder, RegEncoder, Decoder

if __name__ == '__main__':
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='CGVAE')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--split_ratio', type=float, default=0.7)
    parser.add_argument('--num_val', type=float, default=0.2)
    parser.add_argument('--num_test', type=float, default=0.2)
    parser.add_argument('--neg_sample_ratio', type=float, default=1)
    parser.add_argument('--use_edge_for_predict', type=str, default='combined')
    # model train arguments
    parser.add_argument('--layer_type', type=str, default='GATConv')
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--out_channels', type=int, default=32)
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--regularization', type=float, default=2)
    parser.add_argument('--false_pos_edge_ratio', type=float, default=0.1)
    # other arguments
    parser.add_argument('--results', type=str, default='results/results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # create logger
    logging.basicConfig(level=logging.INFO)

    pyg.seed.seed_everything(args.seed)

    # initalize dataloader
    data = cgvae.data_transform.get_data('data/', args.dataset,
                                         mask_ratio=args.split_ratio,
                                         num_val=args.num_val,
                                         num_test=args.num_test,
                                         neg_sample_ratio=args.neg_sample_ratio,
                                         false_pos_edge_ratio=args.false_pos_edge_ratio)

    # count run time from here
    time_start = time.time()

    if args.layer_type =='GCNConv':
        conv_layer = GCNConv
    elif args.layer_type == 'SAGEConv':
        conv_layer = SAGEConv
    else:
        conv_layer = GATConv

    reg_encoder = RegEncoder(conv_layer=conv_layer, hidden_size=args.out_channels * 2, latent_size=args.out_channels)
    recon_encoder = ReconEncoder(conv_layer=conv_layer, hidden_size=args.out_channels * 2, latent_size=args.out_channels)
    num_classes = data['output'].y.max().item() + 1
    classifier = Decoder(input_dim=args.out_channels, hidden_dim=args.out_channels * 2, output_dim=num_classes)

    cgvae_net, val_loss = cgvae.cgvae_train(
        device=args.device,
        data=data,
        classifer=classifier,
        reg_encoder=reg_encoder,
        recon_encoder=recon_encoder,
        out_channels=args.out_channels,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        model_path=osp.join('checkpoints', str(args.seed), 'cgvae_net.pth'),
        regularization=args.regularization,
        split_ratio=args.split_ratio,
        neg_sample_ratio=args.neg_sample_ratio
    )

    end_time = time.time()
    execution_time = end_time - time_start

    accuracy = cgvae.cgvae_model.test(cgvae_net, data)
    print(f'seed: {args.seed}, accuracy: {accuracy}')

    # Create a dictionary with the data you want to save
    data = {
        'dataset': args.dataset,
        'split_ratio': args.split_ratio,
        'use_edge_for_predict': args.use_edge_for_predict,
        'seed': args.seed,
        'val_loss': round(val_loss.item(), 4),
        'accuracy': round(accuracy, 4),
        # 'AUC': round(auc, 4),
        # 'AP': round(ap, 4),
        'learning_rate': args.learning_rate,
        'regularization': args.regularization,
        'neg_sample_ratio': args.neg_sample_ratio,
        'false_pos_edge_ratio': args.false_pos_edge_ratio,
        'execution_time': round(execution_time, 2),
        'time_stamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Read the existing data
    with open(args.results, 'a') as f:
        f.write('\n')
        json.dump(data, f)

