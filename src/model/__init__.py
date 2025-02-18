# add all modules
from .baseline import train as baseline_train
from .cgvae_model import train as cgvae_train
from .cgvae_model import test as cgvae_test
from .cgvae_model import CGVAE
from .baseline import BaselineNet