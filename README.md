# Welcome to LSC-GNN: Latent Space Constraint Graph Neural Network

# install
The package can be easily installed using pip using dependency restriction in pyproject.toml file (you don't have to worry about details). 
```
pip install .
```

`uv` is a powerful dependency management tool (recommended) that streamline the installation, check https://docs.astral.sh/uv/getting-started/installation/ for details.

After you have installed `uv`:
```
uv sync
```

# Run experiment

## Run a single experiment

We manage hyperparameters using `hydra`, please check `config/homogeneous/config.yaml` to see what experiment settings and hyparameters are available. You can simply override default settings using the format below.
```
python main.py -m OVERRIDED_ARGS=NEW_VALUE
```

For example:
```
python main.py -m data.dataset=Cora,CiteSeer,PubMed 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.05)' \
model.model_type=LSCGNN \
 model.use_edge_for_predict=regularization,target model.layer_type=GATConv, GCNConv \
results='results/results_baseline.json'
```
You may notice that the some arguments are given multiple values, and it is supported by hydra to allow run experiment in grid search manner.

## Run batch of experiment
```
bash multi_run.sh
```

# Configured experiments
The experiemnt setting use in the paper.

data.target_ratio: the proportion of nodes in the target graph, e.g. 0.5 means 50% of the nodes is randomly assigned to the target graph and the rest is used for regularization

data.perturb_rate: the additional proportion of false-positive nodes added to the target graph, e.g., 0.1 means additional 10% of number of existing edges are added.

data.perturb_type: Random, Mettack (gradient-based adversarial attack)

model.model_type: Jaccard, LSCGNN, GCNSVD, ProGNN. We use deeprobust's implementation of GCNSVD and ProGNN here. 

train.regularization: the strength of constraint from the external knowledge. When use it for the first time, we need to try a wide range of it to get the appropriate one. This only apply to model.model_type=LSCGNN or Jaccard

model.use_edge_for_predict: the actual edge used for prediction, `full`: all edges are used. `target`: only edges in the target graph. `regularization`: only edges in the regularization graph. `full` is used by default, even regularization is done through edges in reg graph, it is beneficial to use all information at the prediction stage. `target` and `regularization` options are used for ablation analysis here.

model.layer_type: GATConv and GCNConv decide which neighbor aggregation mechanism used, it doesn't apply when model.mode_type is `GCNSVD` or `ProGNN`. It is possible to use different aggregation mechanism for regularization encoder and full encoder, but additional adjustment need to be made in the source code.

Jaccard is a preprocessing approach, it can be combined with other methods. Jaccard is implemented in the following approach: 1. when Jaccard is used with regularization and mode.layer_type is GATConv it is Jaccard+LSCGNN.

When model.model_type is set as LSCGNN without regularization and model.use_edge_for_predict is `full`, it is classical GCN and GAT model (depdending on model.layer_type)

When model.model_type is set to `GCNSVD` and `ProGNN`, it only take model.use_edge_for_predict=full no matter what value is provided, which reflect performance of robust GNN approach used directly on the full graph (target + regularization) without applying different weights.
```
EPOCHS=1
PERTURB_TYPE = Random # or Mettack

# two version of LSCGNN: original LSCGNN and Jaccard+LSCGNN (Jaccard is a preprocessing step)
python main.py -m data.dataset=Cora,CiteSeer,Facebook,PPI 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.1)' data.perturb_type=$PERTURB_TYPE \
model.model_type=LSCGNN,Jaccard \
'train.regularization=range(0, 21, 1)' model.use_edge_for_predict=full model.layer_type=GATConv \
results='results/temp/results_LSC.json' train.num_epochs=$EPOCHS

# use_edge_for_predict: 'full' is used for GAT and GCN baseline comparison, and 'regularization' and 'target' are used for ablation study
python main.py -m data.dataset=Cora,CiteSeer,Facebook,PPI 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.1)' data.perturb_type=$PERTURB_TYPE \
model.model_type=LSCGNN \
 model.use_edge_for_predict=full,regularization,target model.layer_type=GATConv,GCNConv \
results='results/temp/results_baseline.json' train.num_epochs=$EPOCHS

# for robust learning comparison, model.use_edge_for_predict is set automatically to "full" no matter what you set
# because those model are suppose to be robust to perturbation when provide full edge information
python main.py -m data.dataset=Cora,CiteSeer,Facebook,PPI 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.1)' data.perturb_type=PERTURB_TYPE \
model.model_type=Jaccard,GCNSVD,ProGNN \
model.use_edge_for_predict=full model.layer_type=GCNConv \
results='results/temp/results_baseline.json' train.num_epochs=$EPOCHS
```


# use docker image
In the project root directory, execute command below to build a docker image
```
docker build --platform linux/amd64 -t springlight123/gnn-lsc -f docker/Dockerfile .
```

or you can download docker image directly from `https://hub.docker.com/repository/docker/springlight123/lsc-gnn`

The docker image is constructed to use as infrastructure for running code in this project, to use it, move to the project root directory and run the following command. You should replace the `NEW_COMMAND` to the actual command you want to run. Docker image is also useful for conducting experiment in high-computing cluster. For using this image in Kubernetes cluster, please check `docker/train.yaml`. 
```
docker run -it --rm --platform linux/amd64 -v $(pwd):/home springlight123/gnn-lsc [NEW_COMMAND]
```

# Please cite if you find this project helpful
```
@misc{gu2025robustlearningnoisygraphs,
      title={Robust Learning on Noisy Graphs via Latent Space Constraints with External Knowledge}, 
      author={Chunhui Gu and Mohammad Sadegh Nasr and James P. Long and Kim-Anh Do and Ehsan Irajizad},
      year={2025},
      eprint={2507.05540},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.05540}, 
}
```

paper: https://arxiv.org/abs/2507.05540



# Additional notes: 

pytorch and some packages are platform and CPU/GPU dependent . We have tried the best to manage those dependency. But in case when conflict happens or different version is wanted. 
Using the appropriate version based on platform and cuda version
```
uv pip install "pytorch==2.8.0"
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```