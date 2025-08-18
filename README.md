# Welcome to CGVAE: conditional graph variational autoencoder

# install
The package can be easily installed using pip using dependency restriction in pyproject.toml file (you don't have to worry about details). 
```azure
pip install .
```

recommended approach
```
uv sync
```

install pytorch and additional platform-dependent packages.
Using the appropriate version based on platform and cuda version
```
uv pip install "pytorch==2.8.0"
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

