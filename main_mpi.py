import os
import pickle

import hydra
import torch
import torch_geometric as pyg
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf
import  src.model.cgvae_model_hetero as hetero_cgvae
import pytorch_lightning as pl
from src.model.data.hetero_data_module import HeteroDataModule


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):

    pl.seed_everything(36) # fix train/val/test split, negative sampling, and false positive edges

    # load test edge
    # try to use protein-interaction from KEGG as test and STRING for training, regularization has better improvement
    with open('data/edge_index_KEGG.pkl', 'rb') as f:
        test_edge_index = pickle.load(f)


    # try to remove PPI from kegg data
    dm = HeteroDataModule(
        data_path= 'data/pyg_graph_with_string_data_cooccurence.pkl',
        target_node_type='gene',
        target_edge_type=('gene', 'to', 'gene'),
        num_val=cfg.dataset.num_val,
        num_test=cfg.dataset.num_test,
        test_edge_list=test_edge_index,
        neg_sample_ratio=cfg.dataset.neg_sample_ratio,
        batch_size=cfg.train.batch_size,
        target_only_in_recon=cfg.dataset.target_only_in_recon,
        # add_noisy_edges=0.2
    )

    dm.prepare_data()
    dm.setup()

    # todo: add protein-interaction from kegg as reg_graph
    # todo: option 2 add false interaction to kegg data

    model = hetero_cgvae.HeteroCGVAELightning(
                                              hidden_size=2 * cfg.model.out_channels,
                                              latent_size=cfg.model.out_channels,
                                              full_graph_metadata=dm.full_graph_metadata,
                                              reg_graph_metadata=dm.reg_graph_metadata,
                                              target_node_type='gene',
                                              target_edge_type=('gene', 'to', 'gene'),
                                              learning_rate=cfg.train.learning_rate,
                                              regularization=cfg.train.regularization,
                                              neg_sample_ratio=cfg.dataset.neg_sample_ratio)


    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=cfg.other.ckpt_path, filename='best_model', save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=cfg.train.early_stop_patience, mode='min')

    # create logger
    # logger = TensorBoardLogger(default_hp_metric=False, save_dir='lightning_logs', name=cfg.other.log_dir)
    logger = MLFlowLogger(experiment_name="lightning_logs", run_name=cfg.other.log_dir, tracking_uri="file:./ml-runs")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    def flatten_two_level_dict(d):
        flattened_dict = {}
        for outer_key, inner_dict in d.items():
            for inner_key, value in inner_dict.items():
                flattened_dict[inner_key] = value
        return flattened_dict
    cfg_dict = flatten_two_level_dict(cfg_dict)

    logger.log_hyperparams(cfg_dict)
    # logger.log_hyperparams({"seed": cfg.other.seed, 'target_only_in_recon': cfg.dataset.target_only_in_recon,
    #                         'neg_sample_ratio': cfg.dataset.neg_sample_ratio,
    #                         'regularization': cfg.train.regularization,
    #                         'learning_rate': cfg.train.learning_rate,
    #                         'batch_size': cfg.train.batch_size,
    #                         'num_val': cfg.dataset.num_val,
    #                         'num_test': cfg.dataset.num_test,
    #                         'early_stop_patience': cfg.train.early_stop_patience,
    #                         'out_channels': cfg.model.out_channels})

    trainer = pl.Trainer(max_epochs=cfg.train.num_epochs,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         accelerator=  'gpu' if torch.cuda.is_available() else 'cpu',
                         logger=logger,
                         # fast_dev_run=True
                         )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    main()
