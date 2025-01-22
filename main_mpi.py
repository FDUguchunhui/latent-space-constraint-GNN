import pickle
import torch
import torch_geometric as pyg
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import  src.cgvae.cgvae_model_hetero as hetero_cgvae
import argparse
import logging
import torch_geometric.transforms as T
import pytorch_lightning as pl

from src.cgvae.data.hetero_data_module import FullDataLoader, HeteroDataModule

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
    parser.add_argument('--out_channels', type=int, default=32)
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--early_stop_patience', type=int, default=np.Inf)
    parser.add_argument('--regularization', type=float, default=5)
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

    # load test edge

    # try to use protein-interaction from KEGG as test and STRING for training, regularization has better improvement
    with open('data/edge_index_cooccurence.pkl', 'rb') as f:
        test_edge_index = pickle.load(f)

    dm = HeteroDataModule(
        data_path='data/pyg_graph_with_string_data_full.pkl',
        target_edge_type=('gene', 'to', 'gene'),
        num_val=args.num_val,
        num_test=args.num_test,
        test_edge_list=test_edge_index,
        neg_sample_ratio=args.neg_sample_ratio,
    )

    dm.prepare_data()
    dm.setup()

    # todo: add protein-interaction from kegg as reg_graph
    # todo: option 2 add false interaction to kegg data

    model = hetero_cgvae.HeteroCGVAELightning(in_channels=1024,
                                              hidden_size=2 * args.out_channels,
                                              latent_size=args.out_channels,
                                              reg_graph=dm.reg_graph,
                                              full_graph_metadata=dm.full_graph_metadata,
                                              target_node_type='gene',
                                              target_edge_type=('gene', 'to', 'gene'),
                                              learning_rate=args.learning_rate,
                                              regularization=args.regularization,
                                              neg_sample_ratio=args.neg_sample_ratio)


    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=args.model_path, filename='best_model', save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=args.early_stop_patience, mode='min')

    # create logger
    logger = TensorBoardLogger(default_hp_metric=False, save_dir='lightning_logs')
    trainer = pl.Trainer(max_epochs=args.num_epochs,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator='cpu',
                         logger=logger,
                         # fast_dev_run=True
                         )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)