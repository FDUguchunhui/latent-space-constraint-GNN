import argparse
import itertools
import subprocess

import numpy as np

def run_experiment(params):
    (dataset, split_ratio, neg_sample_ratio, false_pos_edge_ratio,
     regularization, add_input_edges_to_output, out_channels, learning_rate, num_epoch, early_stop_patience, seed) = params

    arguments = [
        'python', 'main.py', '--split_ratio', str(split_ratio),
        '--neg_sample_ratio', str(neg_sample_ratio),
        '--dataset', dataset,
        '--false_pos_edge_ratio', str(false_pos_edge_ratio),
        '--regularization', str(regularization),
        '--seed', str(seed),
        '--learning_rate', str(learning_rate),
        '--num_epochs', str(num_epoch),
        '--early_stop_patience', str(early_stop_patience)
    ]

    if add_input_edges_to_output:
        arguments.append('--add_input_edges_to_output')

    subprocess.run(arguments)
    subprocess.run(['sleep', '1'])

if __name__ == '__main__':

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='batch run')
    parser.add_argument('--seed', type=int, default=40)

    args = parser.parse_args()

    # Define the different split_ratio and dataset choices
    split_ratios = [0.7]
    # split_ratios = [1]
    false_pos_edge_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # percentage of true positive edges will be added for false positive edges
    # false_pos_edge_ratios = [0] # percentage of true positive edges will be added for false positive edges
    regularizations = [0, 10, 100, 1000, 1500, 2000]
    # regularizations = [0]
    # !!!!! check this line this should used be used to check without using regularization
    add_input_edges_to_output = [False]
    out_channels = [16]
    neg_sample_ratios = [1]
    learning_rates = [0.005]
    num_epochs = [300]
    # early_stop_patience = [np.iinfo(np.int32).max]
    early_stop_patience = [1000]
    datasets = ['Cora']

    # Iterate over the choices
    # Create a list of all parameter combinations
    param_combinations = list(itertools.product(
        datasets, split_ratios, neg_sample_ratios, false_pos_edge_ratios,
        regularizations, add_input_edges_to_output, out_channels, learning_rates, num_epochs, early_stop_patience
    ))

    # Append seed increments for each experiment
    experiments = [(params + (args.seed + i,)) for i in range(1) for params in param_combinations]

    # Run the experiments
    for experiment in experiments:
        run_experiment(experiment)



#%%
