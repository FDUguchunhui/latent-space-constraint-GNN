import argparse
import subprocess

if __name__ == '__main__':

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='batch run')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Define the different split_ratio and dataset choices
    # split from 0.1 to 0.9 by 0.1
    split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    datasets = ['Cora']

    # Iterate over the choices
    for split_ratio in split_ratios:
        for dataset in datasets:
            for i in range(10):  # Run three times
                seed = args.seed + i  # Increment seed by 1 each time
                # Run main.py with the current split_ratio, dataset, and seed
                subprocess.run(['python', 'main.py', '--split_ratio', str(split_ratio), '--dataset', dataset, '--seed', str(seed)])
                # sleep for 1 seconds
                subprocess.run(['sleep', '1'])