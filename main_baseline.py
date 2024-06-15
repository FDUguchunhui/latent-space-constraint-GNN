import time
import json
import torch
import torch_geometric as pyg
import numpy as np
import src.cgvae as cgvae
import argparse
import os.path as osp
import logging

if __name__ == '__main__':
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='CGVAE')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--num_val', type=float, default=0.2)
    parser.add_argument('--num_test', type=float, default=0.3)
    parser.add_argument('--neg_sample_ratio', type=float, default=1)
    # model train arguments
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--out_channels', type=int, default=16)
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--early_stop_patience', type=int, default=np.Inf)
    parser.add_argument('--regularization', type=float, default=0.5)
    parser.add_argument('--false_pos_edge_ratio', type=float, default=1)
    parser.add_argument('--featureless', action='store_true')
    # other arguments
    parser.add_argument('--results', type=str, default='results/results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    pyg.seed.seed_everything(args.seed) # fix train/val/test split, negative sampling, and false positive edges
    # torch.manual_seed(args.seed)
    # create logger
    logging.basicConfig(level=logging.INFO)

    # initalize dataloader
    data = cgvae.data_transform.get_data('data/', args.dataset,
                                         mask_ratio=args.split_ratio,
                                         num_val=args.num_val,
                                         num_test=args.num_test,
                                         neg_sample_ratio=args.neg_sample_ratio,
                                         false_pos_edge_ratio=args.false_pos_edge_ratio,
                                         featureless=args.featureless, )

    # count run time from here
    time_start = time.time()

    baseline_net = cgvae.baseline_train(
        device='cpu',
        data=data,
        num_node_features=data['input'].x.size(1),
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        model_path=osp.join('checkpoints', str(args.seed), 'baseline_net.pth'),
        early_stop_patience=args.early_stop_patience,
        split_ratio=args.split_ratio,
        neg_sample_ratio=args.neg_sample_ratio,
    )

    auc, ap = cgvae.baseline.test(baseline_net, data)
    print(f'seed: {args.seed}, AUC: {auc}, AP: {ap}')
