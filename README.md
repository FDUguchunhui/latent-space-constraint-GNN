# Welcome to CGVAE: conditional graph variational autoencoder

# install
The package can be easily installed using pip using dependency restriction in pyproject.toml file (you don't have to worry about details). 
```azure
pip install .
```

While `pip install .` can take care of most dependency automatically, you may need to install pytorch manually due to complexity of torch depending a lot of factors, such as CPU archtecure, GPU acceleration, and operation system.
install torch on MacOS
```azure
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
For other platforms check https://pytorch.org/get-started/locally/

