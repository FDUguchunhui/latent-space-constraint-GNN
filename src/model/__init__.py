# add all modules
from .baseline import train as baseline_train
from .LSC import train as cgvae_train
from .LSC import test as cgvae_test
from .LSC import LSCGNN
from .baseline import BaselineNet