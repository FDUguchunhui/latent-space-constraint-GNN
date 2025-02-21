python main.py -m data.dataset=Cora,CiteSeer,PubMed 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.05)' \
'train.regularization=range(0, 11, 1)' model.use_edge_for_predict=full model.layer_type=GATConv \
results='results/results_LSC.json'

python main.py -m data.dataset=Cora,CiteSeer,PubMed 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.05)' \
 model.use_edge_for_predict=regularization,target model.layer_type=GATConv \
results='results/results_GAT.json'

python main.py -m data.dataset=Cora,CiteSeer,PubMed 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.05)' \
model.use_edge_for_predict=full,target,regularization model.layer_type=GCNConv
results='results/results_GCNConv.json'

python main.py -m data.dataset=Cora,CiteSeer,PubMed 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.05)' \
model.use_edge_for_predict=full,target,regularization model.layer_type=SAGEConv\
results='results/results_SAGEConv.json'


