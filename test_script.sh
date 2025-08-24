# use this code to test add_false_pos_edge_ratio actually works
python main.py -m data.dataset=Cora \
data.target_ratio=0.9 \
'data.perturb_rate=range(0, 0.35, 0.05)' data.perturb_type=Mettack \
 model.use_edge_for_predict=target model.layer_type=GATConv \
results='results/results_LSC_test.json' verbose=true