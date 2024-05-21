import argparse
import subprocess

if __name__ == '__main__':

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='batch run')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Define the different split_ratio and dataset choices
    # split from 0.1, 0.15, 0.225, 0.3375, 0.5, 0.75, 0.875, each one roughly doubles the previous one
    split_ratios = [0.1, 0.15, 0.225, 0.3375, 0.5, 0.75, 0.875, 1]
    add_false_pos_edges = [True]
    regularizations = [0.001, 0.01, 0.1, 10, 100, 1000]
    neg_sample_ratios = [1]
    learning_rates = [0.01]
    num_epochs = [1000]
    early_stop_patience = [100]
    datasets = ['Cora']

    # Iterate over the choices
    for dataset in datasets:
        for split_ratio in split_ratios:
            for neg_sample_ratio in neg_sample_ratios:
                for add_false_pos_edge in add_false_pos_edges:
                    for regularization in regularizations:
                        for i in range(3):
                            seed = args.seed + i  # Increment seed by 1 each time
                            # Run main.py with the current split_ratio, dataset, and seed
                            subprocess.run(['python', 'main.py', '--split_ratio', str(split_ratio),
                                            '--neg_sample_ratio', str(neg_sample_ratio),
                                            '--dataset', dataset,
                                            '--add_false_pos_edge' if add_false_pos_edge else '',
                                            '--regularization', str(regularization),
                                            '--seed', str(seed),
                                            '--learning_rate', str(learning_rates[0]),
                                            '--num_epochs', str(num_epochs[0]),
                                            '--early_stop_patience', str(early_stop_patience[0])
                                            ])
                            # sleep for 1 seconds
                            subprocess.run(['sleep', '1'])