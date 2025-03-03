from pytorch_lightning.cli import LightningCLI
from src.data.hetero_data_module import HeteroDataModule
from src.model.LSCGNN_hetero import HeteroCGVAELightning

if __name__ == '__main__':
    cli = LightningCLI(
        model_class=HeteroCGVAELightning,
        datamodule_class=HeteroDataModule
    )