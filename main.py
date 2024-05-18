import copy
import time
import json
import torch
import torch_geometric as pyg
import numpy as np
import torch_geometric.loader
import  src.cgvae as cgvae
import argparse
import os.path as osp
import logging

if __name__ == '__main__':
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='CGVAE')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--num_val', type=float, default=0.1)
    parser.add_argument('--num_test', type=float, default=0.2)
    parser.add_argument('--neg_edge_ratio', type=float, default=1)
    # model train arguments
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=16)
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--regularization', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    # other arguments
    parser.add_argument('--results', type=str, default='results/results.json')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    pyg.seed.seed_everything(args.seed)

    # create logger
    logging.basicConfig(level=logging.INFO)

    # initalize dataloader
    dataloader, dataset_size = cgvae.data_transform.get_data('data/', args.dataset,
                                                             mask_ratio=args.split_ratio,
                                                             num_val=args.num_val,
                                                             num_test=args.num_test,
                                                             neg_edge_ratio=args.neg_edge_ratio)

    # count run time from here
    time_start = time.time()

    baseline_net = cgvae.baseline_train(
        device='cpu',
        dataloader=dataloader,
        num_node_features=next(iter(dataloader))['input'].x.size(1),
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        model_path=osp.join('checkpoints', str(args.seed), 'baseline_net.pth'),
        early_stop_patience=args.early_stop_patience
    )

    cgvae_net = cgvae.cgvae_train(
        device=args.device,
        dataloader=dataloader,
        num_node_features=next(iter(dataloader))['input'].x.size(1),
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        pre_trained_baseline_net=baseline_net,
        model_path=osp.join('checkpoints', str(args.seed), 'cgvae_net.pth'),
        early_stop_patience=args.early_stop_patience,
        regularization=args.regularization
    )

    end_time = time.time()
    execution_time = end_time - time_start

    dataloader, dataset_size = cgvae.get_data('data/', args.dataset,
                                                             mask_ratio=args.split_ratio,
                                                             num_val=args.num_val, num_test=args.num_test,
                                                             neg_edge_ratio=args.neg_edge_ratio)

    auc, ap = cgvae.cgvae_model.test(cgvae_net, dataloader)
    print(f'seed: {args.seed}, AUC: {auc}, AP: {ap}')

    # Create a dictionary with the data you want to save
    data = {
        'dataset': args.dataset,
        'split_ratio': args.split_ratio,
        'seed': args.seed,
        'AUC': round(auc, 4),
        'AP': round(ap, 4),
        'execution_time': round(execution_time, 2)
    }

    # Read the existing data
    with open(args.results, 'a') as f:
        f.write('\n')
        json.dump(data, f)


