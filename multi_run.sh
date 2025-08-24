EPOCHS=1
PERTURB=Random

# two version of LSCGNN: original LSCGNN and Jaccard+LSCGNN (Jaccard is a preprocessing step)
python main.py -m data.dataset=Cora,CiteSeer,Facebook,PPI 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.1)' data.perturb_type=$PERTURB \
model.model_type=LSCGNN,Jaccard \
'train.regularization=range(0, 21, 1)' model.use_edge_for_predict=full model.layer_type=GATConv \
results="results/temp/results_${PERTURB}_LSC.json" train.num_epochs=$EPOCHS

# use_edge_for_predict: 'full' is used for GAT and GCN baseline comparison, and 'regularization' and 'target' are used for ablation study
python main.py -m data.dataset=Cora,CiteSeer,Facebook,PPI 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.1)' data.perturb_type=$PERTURB \
model.model_type=LSCGNN \
 model.use_edge_for_predict=full,regularization,target model.layer_type=GATConv,GCNConv \
results="results/temp/results_${PERTURB}_baseline.json" train.num_epochs=$EPOCHS

# for robust learning comparison, model.use_edge_for_predict is set automatically to "full" no matter what you set
# because those model are suppose to be robust to perturbation when provide full edge information
python main.py -m data.dataset=Cora,CiteSeer,Facebook,PPI 'seed=range(1, 11, 1)' \
data.target_ratio=0.5,0.7,0.9 \
'data.perturb_rate=range(0, 0.35, 0.1)' data.perturb_type=$PERTURB \
model.model_type=Jaccard,GCNSVD,ProGNN \
model.layer_type=GCNConv \
results="results/temp/results_${PERTURB}_baseline.json" train.num_epochs=$EPOCHS


#python main.py data.dataset=PubMed \
#data.target_ratio=0.5 \
#data.perturb_rate=0.3 \
#model.model_type=GCNJaccard \
# model.use_edge_for_predict=regularization,target model.layer_type=GCNConv \
#data.perturb_type='Mettack' \
#results='results/results.json'