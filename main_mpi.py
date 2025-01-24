import os
import pickle

import hydra
import torch
import torch_geometric as pyg
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
import  src.cgvae.cgvae_model_hetero as hetero_cgvae
import argparse
import pytorch_lightning as pl
from src.cgvae.data.hetero_data_module import HeteroDataModule


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):

    pyg.seed.seed_everything(cfg.other.seed) # fix train/val/test split, negative sampling, and false positive edges

    # load test edge
    # try to use protein-interaction from KEGG as test and STRING for training, regularization has better improvement
    with open('data/edge_index_cooccurence.pkl', 'rb') as f:
        test_edge_index = pickle.load(f)

    dm = HeteroDataModule(
        data_path= 'data/pyg_graph_with_string_data_full.pkl',
        target_edge_type=('gene', 'to', 'gene'),
        num_val=cfg.dataset.num_val,
        num_test=cfg.dataset.num_test,
        test_edge_list=test_edge_index,
        neg_sample_ratio=cfg.dataset.neg_sample_ratio,
    )

    dm.prepare_data()
    dm.setup()

    # todo: add protein-interaction from kegg as reg_graph
    # todo: option 2 add false interaction to kegg data

    model = hetero_cgvae.HeteroCGVAELightning(in_channels=1024,
                                              hidden_size=2 * cfg.model.out_channels,
                                              latent_size=cfg.model.out_channels,
                                              reg_graph=dm.reg_graph,
                                              full_graph_metadata=dm.full_graph_metadata,
                                              target_node_type='gene',
                                              target_edge_type=('gene', 'to', 'gene'),
                                              learning_rate=cfg.train.learning_rate,
                                              regularization=cfg.train.regularization,
                                              neg_sample_ratio=cfg.dataset.neg_sample_ratio)


    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=cfg.other.model_path, filename='best_model', save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=cfg.train.early_stop_patience, mode='min')

    # create logger
    logger = TensorBoardLogger(default_hp_metric=False, save_dir=cfg.other.log_dir)
    logger.log_hyperparams({'seed': cfg.other.seed})

    trainer = pl.Trainer(max_epochs=cfg.train.num_epochs,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator='cpu',
                         logger=logger,
                         # fast_dev_run=True
                         )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == '__main__':
    main()
