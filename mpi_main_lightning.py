import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from src.model.data.hetero_data_module import HeteroDataModule
from src.model.cgvae_model_hetero import HeteroCGVAELightning

if __name__ == '__main__':
    cli = LightningCLI(
        model_class=HeteroCGVAELightning,
        datamodule_class=HeteroDataModule
    )