import argparse
import subprocess

if __name__ == '__main__':

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='batch run')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Define the different split_ratio and dataset choices
    # split from 0.1, 0.15, 0.225, 0.3375, 0.5, 0.75, 0.875, each one roughly doubles the previous one
    split_ratios = [0.1, 0.15, 0.225, 0.3375, 0.5, 0.75, 0.875]
    datasets = ['PubMed']
    # try regularization 0.1 0.3, 0.5, 1, 2
    regularization = [0.1, 0.3, 0.5, 1, 2]
    # build grid list using split_ratio and dataset

    # Iterate over the choices
    for dataset in datasets:
        for split_ratio in split_ratios:
            for i in range(3):  # Run three times
                seed = args.seed + i  # Increment seed by 1 each time
                # Run main.py with the current split_ratio, dataset, and seed
                subprocess.run(['python', 'main_vgae.py', '--dataset', dataset, '--seed', str(seed)])
                # sleep for 1 seconds
                subprocess.run(['sleep', '1'])