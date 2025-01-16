import pickle
import torch
import torch_geometric as pyg
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import  src.cgvae.cgvae_model_hetero as hetero_cgvae
import argparse
import logging
import torch_geometric.transforms as T
import pytorch_lightning as pl

from src.cgvae.utils import FullDataLoader

if __name__ == '__main__':
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='CGVAE')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--num_val', type=float, default=0.1)
    parser.add_argument('--num_test', type=float, default=0.2)
    parser.add_argument('--neg_sample_ratio', type=float, default=1)
    parser.add_argument('--add_input_edges_to_output', action='store_true')
    # model train arguments
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--out_channels', type=int, default=16)
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--early_stop_patience', type=int, default=np.Inf)
    parser.add_argument('--regularization', type=float, default=1000)
    parser.add_argument('--false_pos_edge_ratio', type=float, default=1.0)
    parser.add_argument('--featureless', action='store_true')
    # other arguments
    parser.add_argument('--results', type=str, default='results/hetero_results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    pyg.seed.seed_everything(args.seed) # fix train/val/test split, negative sampling, and false positive edges
    # torch.manual_seed(args.seed)
    # create logger
    logging.basicConfig(level=logging.INFO)

    # load dataset using pickle
    with open('data/pyg_graph.pkl', 'rb') as f:
        data = pickle.load(f)

    transformer = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        neg_sampling_ratio=1.0,
        split_labels=True,
        add_negative_train_samples=False,
        edge_types=("gene", "to", "gene")
    )

    train_data, val_data, test_data = transformer(data)
    data = {}
    data['train'] = train_data
    data['val'] = val_data
    data['test'] = test_data

    reg_graph = data['train'].clone()
    del reg_graph[('gene', 'to', 'gene')]

    model = hetero_cgvae.HeteroCGVAELightning(in_channels=1024,
                                              hidden_size=2 * args.out_channels,
                                              latent_size=args.out_channels,
                                              reg_graph=reg_graph,
                                              full_graph_metadata=data['train'].metadata(),
                                              target_node_type='gene',
                                              target_edge_type=('gene', 'to', 'gene'),
                                              learning_rate=args.learning_rate,
                                              regularization=args.regularization,
                                              neg_sample_ratio=args.neg_sample_ratio)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=args.model_path, filename='best_model', save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=args.early_stop_patience, mode='min')

    trainer = pl.Trainer(max_epochs=args.num_epochs,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator='cpu',
                         # fast_dev_run=True
                         )
    trainer.fit(model, train_dataloaders=FullDataLoader(data['train']), val_dataloaders=FullDataLoader(data['val']))
    trainer.test(model, dataloaders=FullDataLoader(data['test']))

    # # Create a dictionary with the data you want to save
    # data = {
    #     'dataset': args.dataset,
    #     'split_ratio': args.split_ratio,
    #     'seed': args.seed,
    #     'best_epochs': best_epoch,
    #     'val_best_loss': round(val_best_loss.item(), 4),
    #     'AUC': round(auc, 4),
    #     'AP': round(ap, 4),
    #     'learning_rate': args.learning_rate,
    #     'regularization': args.regularization,
    #     'neg_sample_ratio': args.neg_sample_ratio,
    #     'false_pos_edge_ratio': args.false_pos_edge_ratio,
    #     'add_input_edges_to_output': args.add_input_edges_to_output,
    #     'execution_time': round(execution_time, 2),
    #     'time_stamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    # }

    # # Read the existing data
    # with open(args.results, 'a') as f:
    #     f.write('\n')
    #     json.dump(data, f)

