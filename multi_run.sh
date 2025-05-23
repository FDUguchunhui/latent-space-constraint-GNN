#python main.py -m data.dataset=Cora,CiteSeer 'seed=range(1, 11, 1)' \
#data.target_ratio=0.5,0.7,0.9 \
#'data.perturb_rate=range(0, 0.35, 0.05)' \
#model.model_type=LSCGNN \
#'train.regularization=range(0, 11, 1)' model.use_edge_for_predict=full model.layer_type=GATConv \
#results='results/results_LSC.json'
#
#python main.py -m data.dataset=PubMed 'seed=range(1, 11, 1)' \
#data.target_ratio=0.5,0.7,0.9 \
#'data.perturb_rate=range(0, 0.35, 0.05)' \
#model.model_type=LSCGNN \
#'train.regularization=range(0, 21, 1)' model.use_edge_for_predict=full model.layer_type=GATConv \
#results='results/results_LSC.json'

python main.py -m data.dataset=Cora,CiteSeer,PubMed 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.05)' \
model.model_type=LSCGNN \
 model.use_edge_for_predict=regularization,target model.layer_type=GATConv, GCNConv \
results='results/results_baseline.json'

python main.py -m data.dataset=Cora,CiteSeer,PubMed 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.05)' \
model.model_type=GCNJaccard, GCNSVD, ProGNN \
 model.use_edge_for_predict=regularization,target model.layer_type=GCNConv \
results='results/results_baseline.json'


#python main.py data.dataset=PubMed \
#data.target_ratio=0.5 \
#data.perturb_rate=0.3 \
#model.model_type=GCNJaccard \
# model.use_edge_for_predict=regularization,target model.layer_type=GCNConv \
#data.perturb_type='Mettack' \
#results='results/results.json'